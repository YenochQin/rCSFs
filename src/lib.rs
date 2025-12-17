use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyIOError};
use pyo3::types::{PyDict, PyDictMethods};
use std::path::Path;
use std::io::{BufRead, BufReader};

mod csfs_conversion;

#[pymodule]
fn _csfs_loader(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(convert_csfs_to_parquet, m)?)?;
    m.add_function(wrap_pyfunction!(get_parquet_metadata, m)?)?;
    m.add_class::<CSFProcessor>()?;

    // Add the new functions defined in .pyi
    m.add_function(wrap_pyfunction!(convert_csfs, m)?)?;
    m.add_function(wrap_pyfunction!(csfs_header, m)?)?;

    Ok(())
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
/// - total_lines: 总行数
/// - csf_count: CSF 数量
/// - truncated_count: 截断行数
/// - input_file: 输入文件路径
/// - output_file: 输出文件路径
/// - header_file: 头部文件路径
#[pyfunction]
fn convert_csfs_to_parquet(
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
        Ok(_) => {
            // 创建结果字典
            let stats = PyDict::new(py);
            stats.set_item("success", true)?;
            stats.set_item("input_file", &input_path)?;
            stats.set_item("output_file", &output_path)?;
            stats.set_item("max_line_len", max_line_len)?;
            stats.set_item("chunk_size", chunk_size)?;

            // 尝试读取头部文件路径
            let header_path = Path::new(&output_path).with_extension("headers.txt");
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

/// 获取 Parquet 文件基本信息（仅元数据）
///
/// 参数:
/// - input_path: Parquet 文件路径
///
/// 返回:
/// 包含基本文件信息的字典
#[pyfunction]
fn get_parquet_metadata(
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

    /// 转换 CSF 文件
    fn convert(&self, py: Python, input_path: String, output_path: String) -> PyResult<pyo3::Py<pyo3::PyAny>> {
        convert_csfs_to_parquet(
            py,
            input_path,
            output_path,
            Some(self.max_line_len),
            Some(self.chunk_size),
        )
    }

    /// 获取 Parquet 文件元数据
    fn get_metadata(&self, py: Python, input_path: String) -> PyResult<pyo3::Py<pyo3::PyAny>> {
        get_parquet_metadata(py, input_path)
    }
}

/// 简化的 CSF 转换函数（与 .pyi 中定义的接口匹配）
#[pyfunction]
fn convert_csfs(
    py: Python,
    input_path: String,
    output_path: String,
    max_line_len: Option<usize>,
    chunk_size: Option<usize>,
) -> PyResult<pyo3::Py<pyo3::PyAny>> {
    convert_csfs_to_parquet(py, input_path, output_path, max_line_len, chunk_size)
}

/// 提取 CSF 文件头部的函数
#[pyfunction]
fn csfs_header(
    py: Python,
    input_path: String,
) -> PyResult<pyo3::Py<pyo3::PyAny>> {
    // 读取文件头部（前5行）
    match std::fs::File::open(&input_path) {
        Ok(file) => {
            let reader = BufReader::new(file);
            let mut lines = Vec::new();

            for (i, line_result) in reader.lines().take(5).enumerate() {
                match line_result {
                    Ok(line) => lines.push(line),
                    Err(e) => {
                        return Err(PyIOError::new_err(
                            format!("Error reading line {}: {}", i + 1, e)
                        ));
                    }
                }
            }

            // 创建结果字典
            let dict = PyDict::new(py);
            dict.set_item("header_lines", lines.len())?;
            dict.set_item("file_path", &input_path)?;

            // 添加每一行到结果中
            for (i, line) in lines.iter().enumerate() {
                dict.set_item(format!("line{}", i + 1), line)?;
            }

            Ok(dict.into())
        }
        Err(e) => {
            Err(PyIOError::new_err(
                format!("Failed to open file {}: {}", input_path, e)
            ))
        }
    }
}
