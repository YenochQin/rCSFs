use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyIOError};
use pyo3::types::{PyDict, PyDictMethods};
use std::path::Path;

// Import num_cpus for parallel processing
extern crate num_cpus;

mod csfs_conversion;
mod csfs_descriptor;

#[pymodule]
fn _rcsfs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(convert_csfs, m)?)?;
    m.add_function(wrap_pyfunction!(convert_csfs_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(get_parquet_info, m)?)?;
    m.add_class::<CSFProcessor>()?;

    // Register CSF descriptor module
    csfs_descriptor::register_descriptor_module(m)?;

    Ok(())
}


/// 获取 Parquet 文件基本信息和元数据
///
/// 参数:
/// - input_path: Parquet 文件路径
///
/// 返回:
/// 包含文件信息和元数据的字典
#[pyfunction]
fn get_parquet_info(
    py: Python,
    input_path: String,
) -> PyResult<pyo3::Py<pyo3::PyAny>> {
    py.detach(|| {
        csfs_conversion::get_parquet_metadata(Path::new(&input_path))
            .map_err(|e| PyIOError::new_err(format!("获取 Parquet 文件信息失败: {}", e)))
    })
}

/// CSF 文件处理器类，提供面向对象的接口
#[pyclass]
#[derive(Clone)]
struct CSFProcessor {
    max_line_len: usize,
    chunk_size: usize,
}

#[pymethods]
impl CSFProcessor {
    /// 创建新的 CSF 处理器实例
    #[new]
    #[pyo3(signature = (max_line_len=256, chunk_size=30000))]
    fn new(max_line_len: Option<usize>, chunk_size: Option<usize>) -> PyResult<Self> {
        let max_line_len = max_line_len.unwrap_or(256);
        let chunk_size = chunk_size.unwrap_or(30000);

        if max_line_len == 0 {
            return Err(PyValueError::new_err("max_line_len 必须大于 0"));
        }
        if chunk_size == 0 {
            return Err(PyValueError::new_err("chunk_size 必须大于 0"));
        }

        Ok(CSFProcessor {
            max_line_len,
            chunk_size,
        })
    }

    /// 设置最大行长度
    #[setter]
    fn set_max_line_len(&mut self, value: usize) -> PyResult<()> {
        if value == 0 {
            return Err(PyValueError::new_err("max_line_len 必须大于 0"));
        }
        self.max_line_len = value;
        Ok(())
    }

    /// 设置批处理大小
    #[setter]
    fn set_chunk_size(&mut self, value: usize) -> PyResult<()> {
        if value == 0 {
            return Err(PyValueError::new_err("chunk_size 必须大于 0"));
        }
        self.chunk_size = value;
        Ok(())
    }

    /// 获取当前配置
    fn get_config(&self, py: Python) -> PyResult<pyo3::Py<pyo3::PyAny>> {
        let config = PyDict::new(py);
        config.set_item("max_line_len", self.max_line_len)?;
        config.set_item("chunk_size", self.chunk_size)?;
        Ok(config.into())
    }

    /// 转换 CSF 文件（顺序版本）
    fn convert(&self, py: Python, input_path: String, output_path: String) -> PyResult<pyo3::Py<pyo3::PyAny>> {
        convert_csfs(
            py,
            input_path,
            output_path,
            Some(self.max_line_len),
            Some(self.chunk_size),
        )
    }

    /// 转换 CSF 文件（并行版本，适用于大规模数据）
    ///
    /// 参数:
    /// - num_workers: 工作线程数（可选，默认为 CPU 核心数）
    fn convert_parallel(
        &self,
        py: Python,
        input_path: String,
        output_path: String,
        num_workers: Option<usize>
    ) -> PyResult<pyo3::Py<pyo3::PyAny>> {
        convert_csfs_parallel(
            py,
            input_path,
            output_path,
            Some(self.max_line_len),
            Some(self.chunk_size),
            num_workers,
        )
    }

    /// 获取 Parquet 文件信息
    fn get_metadata(&self, py: Python, input_path: String) -> PyResult<pyo3::Py<pyo3::PyAny>> {
        get_parquet_info(py, input_path)
    }
}

/// 将 CSF 文本文件转换为 Parquet 格式
///
/// 参数:
/// - input_path: 输入 CSF 文件路径
/// - output_path: 输出 Parquet 文件路径
/// - max_line_len: 最大行长度（默认 256）
/// - chunk_size: 批处理大小（默认 30000）
///
/// 返回:
/// 包含转换统计信息的字典:
/// - success: 转换是否成功
/// - csf_count: CSF 数量
/// - total_lines: 总行数
/// - truncated_count: 截断行数
/// - input_file: 输入文件路径
/// - output_file: 输出文件路径
/// - header_file: TOML 头部文件路径
/// - max_line_len: 使用的最大行长度
/// - chunk_size: 使用的批处理大小
/// - error: 错误信息（仅在失败时存在）
///
/// 头部文件格式:
/// 自动生成 [输入文件名前缀]_header.toml 文件，包含:
/// - header_info: 包含 header_lines 子属性
///   - header_lines: 头部行内容列表
/// - conversion_stats: 转换统计信息
#[pyfunction]
fn convert_csfs(
    py: Python,
    input_path: String,
    output_path: String,
    max_line_len: Option<usize>,
    chunk_size: Option<usize>,
) -> PyResult<pyo3::Py<pyo3::PyAny>> {
    // 设置默认参数
    let max_line_len = max_line_len.unwrap_or(256);
    let chunk_size = chunk_size.unwrap_or(30000);

    // 参数验证
    if max_line_len == 0 {
        return Err(PyValueError::new_err("max_line_len 必须大于 0"));
    }
    if chunk_size == 0 {
        return Err(PyValueError::new_err("chunk_size 必须大于 0"));
    }

    // 执行转换
    let result = py.detach(|| {
        csfs_conversion::convert_csfs_to_parquet(
            Path::new(&input_path),
            Path::new(&output_path),
            max_line_len,
            chunk_size,
        )
    });

    match result {
        Ok(conversion_stats) => {
            // 创建结果字典
            let stats = PyDict::new(py);
            stats.set_item("success", true)?;
            stats.set_item("input_file", &input_path)?;
            stats.set_item("output_file", &output_path)?;
            stats.set_item("max_line_len", max_line_len)?;
            stats.set_item("chunk_size", chunk_size)?;
            stats.set_item("csf_count", conversion_stats.csf_count)?;
            stats.set_item("total_lines", conversion_stats.total_lines)?;
            stats.set_item("truncated_count", conversion_stats.truncated_count)?;

            // 尝试读取 [输入文件名前缀]_header.toml 文件路径
            let output_dir = Path::new(&output_path).parent().unwrap_or_else(|| Path::new("."));
            let input_file_stem = Path::new(&input_path).file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("csfs");
            let header_filename = format!("{}_header.toml", input_file_stem);
            let header_path = output_dir.join(header_filename);
            if header_path.exists() {
                stats.set_item("header_file", header_path.to_string_lossy())?;
            }

            Ok(stats.into())
        }
        Err(e) => {
            // 创建错误结果字典
            let stats = PyDict::new(py);
            stats.set_item("success", false)?;
            stats.set_item("error", e.to_string())?;
            Ok(stats.into())
        }
    }
}

/// 将 CSF 文本文件转换为 Parquet 格式（并行版本，适用于大规模数据）
///
/// 参数:
/// - input_path: 输入 CSF 文件路径
/// - output_path: 输出 Parquet 文件路径
/// - max_line_len: 最大行长度（默认 256）
/// - chunk_size: 批处理大小（默认 50000，建议增大以提高并行效率）
/// - num_workers: 工作线程数（默认为 CPU 核心数）
///
/// 返回:
/// 包含转换统计信息的字典:
/// - success: 转换是否成功
/// - csf_count: CSF 数量
/// - total_lines: 总行数
/// - truncated_count: 截断行数
/// - input_file: 输入文件路径
/// - output_file: 输出文件路径
/// - header_file: TOML 头部文件路径
/// - max_line_len: 使用的最大行长度
/// - chunk_size: 使用的批处理大小
/// - num_workers: 使用的工作线程数
/// - error: 错误信息（仅在失败时存在）
///
/// 特性:
/// - 多线程并行处理，显著提升大规模数据处理速度
/// - 保持原始 CSF 顺序，确保输出文件顺序一致
/// - 内存高效，通过有界队列防止内存爆炸
/// - 实时进度监控和统计信息
/// - 自动工作线程数优化（默认使用所有 CPU 核心）
///
/// 推荐设置:
/// - 大规模文件（>10M CSF）: chunk_size=100000, num_workers=8+
/// - 中等规模文件（1-10M CSF）: chunk_size=50000, num_workers=4-8
/// - 小规模文件（<1M CSF）: 使用 convert_csfs() 即可
#[pyfunction]
fn convert_csfs_parallel(
    py: Python,
    input_path: String,
    output_path: String,
    max_line_len: Option<usize>,
    chunk_size: Option<usize>,
    num_workers: Option<usize>,
) -> PyResult<pyo3::Py<pyo3::PyAny>> {
    // 设置默认参数（针对并行处理优化）
    let max_line_len = max_line_len.unwrap_or(256);
    let chunk_size = chunk_size.unwrap_or(50000); // 增大默认 chunk_size
    let num_workers = num_workers.unwrap_or_else(|| num_cpus::get());

    // 参数验证
    if max_line_len == 0 {
        return Err(PyValueError::new_err("max_line_len 必须大于 0"));
    }
    if chunk_size == 0 {
        return Err(PyValueError::new_err("chunk_size 必须大于 0"));
    }
    if num_workers == 0 {
        return Err(PyValueError::new_err("num_workers 必须大于 0"));
    }

    // 执行并行转换
    let result = py.detach(|| {
        csfs_conversion::convert_csfs_to_parquet_parallel(
            Path::new(&input_path),
            Path::new(&output_path),
            max_line_len,
            chunk_size,
            Some(num_workers),
        )
    });

    match result {
        Ok(conversion_stats) => {
            // 创建结果字典
            let stats = PyDict::new(py);
            stats.set_item("success", true)?;
            stats.set_item("input_file", &input_path)?;
            stats.set_item("output_file", &output_path)?;
            stats.set_item("max_line_len", max_line_len)?;
            stats.set_item("chunk_size", chunk_size)?;
            stats.set_item("num_workers", num_workers)?;
            stats.set_item("csf_count", conversion_stats.csf_count)?;
            stats.set_item("total_lines", conversion_stats.total_lines)?;
            stats.set_item("truncated_count", conversion_stats.truncated_count)?;

            // 尝试读取 [输入文件名前缀]_header.toml 文件路径
            let output_dir = Path::new(&output_path).parent().unwrap_or_else(|| Path::new("."));
            let input_file_stem = Path::new(&input_path).file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("csfs");
            let header_filename = format!("{}_header.toml", input_file_stem);
            let header_path = output_dir.join(header_filename);
            if header_path.exists() {
                stats.set_item("header_file", header_path.to_string_lossy())?;
            }

            Ok(stats.into())
        }
        Err(e) => {
            // 创建错误结果字典
            let stats = PyDict::new(py);
            stats.set_item("success", false)?;
            stats.set_item("error", e.to_string())?;
            Ok(stats.into())
        }
    }
}
