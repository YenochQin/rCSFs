"""Build an isolated, instrumented copy of the frozen rcsfgenerate sources."""

import argparse
from pathlib import Path
import re
import shutil
import subprocess

MODULE = """module benchmark_profile
  use iso_fortran_env, only: int64, real64
  implicit none
  integer(int64) :: started(7)=0, elapsed(7)=0, calls(7)=0, bytes(7)=0, read_bytes(7)=0
contains
  subroutine bench_begin(section)
    integer, intent(in) :: section
    call system_clock(started(section))
    calls(section)=calls(section)+1
  end subroutine
  subroutine bench_end(section)
    integer, intent(in) :: section
    integer(int64) :: now
    call system_clock(now)
    elapsed(section)=elapsed(section)+now-started(section)
  end subroutine
  subroutine bench_report()
    integer :: unit, i
    integer(int64) :: rate
    call system_clock(count_rate=rate)
    open(newunit=unit, file='profile.toml', status='new')
    do i=1,7
      write(unit,'(A,I0,A)') '[section_',i,']'
      write(unit,'(A,F20.9)') 'seconds = ',real(elapsed(i),real64)/real(rate,real64)
      write(unit,'(A,I0)') 'calls = ',calls(i)
      write(unit,'(A,I0)') 'record_write_bytes = ',bytes(i)
      write(unit,'(A,I0)') 'record_read_bytes = ',read_bytes(i)
    end do
    close(unit)
  end subroutine
end module
"""


def wrap_calls(text: str, sections: dict[str, int]) -> str:
    """Wrap whole free-form CALL statements, including continuation lines."""
    lines = text.splitlines(keepends=True)
    result = []
    i = 0
    while i < len(lines):
        statement = lines[i]
        i += 1
        if re.match(r"\s*call\b", statement, re.I):
            while statement.rstrip().endswith("&"):
                statement += lines[i]
                i += 1
            name = re.match(r"\s*call\s+(\w+)", statement.replace("&", " "), re.I)
            section = sections.get(name[1].lower()) if name else None
            if section is not None:
                result.append(f"      call bench_begin({section})\n")
                result.append(statement)
                result.append(f"      call bench_end({section})\n")
                continue
        result.append(statement)
    return "".join(result)


def instrument(source: Path) -> None:
    for name, sections in {
        "wrapper.f90": {"rcsfexcitation": 1},
        "jjgen15b.f90": {"blanda": 2, "merge": 5, "rcsfblock": 6, "copy7t9": 7},
        "blanda.f90": {"gen": 3},
    }.items():
        path = source / name
        text = path.read_text()
        if name == "wrapper.f90":
            text = text.replace(
                "program wrapper", "program wrapper\nuse benchmark_profile", 1
            )
        else:
            text = re.sub(
                r"(?im)^(\s*)use ",
                r"\1use benchmark_profile\n      use ",
                text,
                count=1,
            )
        text = wrap_calls(text, sections)
        if name == "jjgen15b.f90":
            text = re.sub(
                r"(?im)^(\s*)stop\s*$", r"\1call bench_report()\n      stop", text
            )
        path.write_text(text)

    path = source / "genb.f90"
    text = path.read_text().replace(
        "      USE kopp1_I", "      USE benchmark_profile\n      USE kopp1_I", 1
    )
    text, count = re.subn(
        r"(?im)^(\s*)CALL KOPP1", r"\1call bench_begin(4)\n\1CALL KOPP1", text
    )
    if count != 20:
        raise ValueError("expected exactly 20 GEN output branches in the frozen source")
    text, count = re.subn(
        r"(?im)^(\s*WRITE \(FIL, 999\) RAD3\(1:(\d+)\))$",
        lambda m: (
            f"{m[1]}\n      bytes(4)=bytes(4)+{3 * int(m[2]) - 1}\n      call bench_end(4)"
        ),
        text,
    )
    if count != 20:
        raise ValueError("expected exactly 20 GEN record writes")
    path.write_text(text)

    path = source / "merge.f90"
    text = path.read_text().replace(
        "      use lika_I", "      use benchmark_profile\n      use lika_I", 1
    )
    text, count = re.subn(
        r"(?im)^(\s*write \((?:utfil|nyfil), 999\) rad\d\d\(1:(stopp[12])\))$",
        lambda m: f"{m[1]}\n      bytes(5)=bytes(5)+{m[2]}+1",
        text,
    )
    if count != 18:
        raise ValueError(f"expected 18 MERGE record writes, got {count}")
    path.write_text(text)

    # GEN and MERGE emit fixed-width record bodies: 9*N, 9*N, 9*N+2.
    # LASA1's first line ends with ')'; LASA2 receives 9*N in stopp.
    # These counters exclude headers and count logical bytes, not device I/O.
    path = source / "lasa1.f90"
    text = path.read_text().replace(
        "      use reada_I", "      use benchmark_profile\n      use reada_I", 1
    )
    text = text.replace(
        "         call reada (rad, pop, skal, slut)",
        "         read_bytes(5)=read_bytes(5)+len_trim(rad)+1\n         call reada (rad, pop, skal, slut)",
    )
    path.write_text(text)
    path = source / "lasa2.f90"
    text = path.read_text().replace(
        "      implicit none", "      use benchmark_profile\n      implicit none", 1
    )
    text = text.replace(
        "         read (fil, 999, end=10) rad2",
        "         read (fil, 999, end=10) rad2\n         read_bytes(5)=read_bytes(5)+stopp+1",
    )
    text = text.replace(
        "         read (fil, 999, end=10) rad3",
        "         read (fil, 999, end=10) rad3\n         read_bytes(5)=read_bytes(5)+stopp+3",
    )
    path.write_text(text)

    path = source / "rcsfblock.f90"
    text = path.read_text().replace(
        "      SUBROUTINE RCSFBLOCK",
        "      SUBROUTINE RCSFBLOCK\n      use benchmark_profile",
        1,
    )
    text, count = re.subn(
        r"(?im)^(\s*WRITE \((?:iunit|19),'\(A\)'\) TRIM\((line[123])\))$",
        lambda m: f"{m[1]}\n      bytes(6)=bytes(6)+len_trim({m[2]})+1",
        text,
    )
    if count != 4:
        raise ValueError(f"expected 4 BLOCK record writes, got {count}")
    text = text.replace(
        "      READ (19, '(A)') LINE3",
        "      READ (19, '(A)') LINE3\n      read_bytes(6)=read_bytes(6)+3*len_trim(line1)+5",
    )
    text = text.replace(
        "  200    READ (iunit, '(A)', END = 9999) line1",
        "  200    READ (iunit, '(A)', END = 9999) line1\n         read_bytes(6)=read_bytes(6)+len_trim(line1)+1",
    )
    path.write_text(text)
    (source / "benchmark_profile.f90").write_text(MODULE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grasp-source", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new directory outside the GRASP checkout",
    )
    parser.add_argument("--optimization", choices=["debug", "release"], default="debug")
    parser.add_argument("--uninstrumented", action="store_true")
    args = parser.parse_args()
    original = (args.grasp_source / "src/appl/rcsfgenerate90").resolve()
    output = args.output.resolve()
    if output.is_relative_to(args.grasp_source.resolve()):
        parser.error("output must be outside the original GRASP checkout")
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    names = re.findall(
        r"^\s+(\S+\.f90)\s*$", (original / "CMakeLists.txt").read_text(), re.M
    )
    if not names:
        raise ValueError("missing rcsfgenerate CMake source list")
    for name in names:
        shutil.copy2(original / name, source / name)
    if not args.uninstrumented:
        instrument(source)
    flags = "-O0 -g" if args.optimization == "debug" else "-O3"
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\nproject(rcsf_benchmark LANGUAGES Fortran)\n"
        "file(GLOB sources CONFIGURE_DEPENDS *.f90)\nadd_executable(rcsfgenerate ${sources})\n"
        f"target_compile_options(rcsfgenerate PRIVATE -fno-automatic -fallow-argument-mismatch {flags})\n"
    )
    subprocess.run(
        ["cmake", "-S", str(source), "-B", str(output / "build")], check=True
    )
    subprocess.run(
        ["cmake", "--build", str(output / "build"), "--parallel", "4"], check=True
    )
    print(output / "build/rcsfgenerate")


if __name__ == "__main__":
    main()
