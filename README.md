# Deep Learning integration on MATLAB Simulink 

This repository contains code and documentation for integrating **machine learning algorithms** into the **plasma digital control system** of the **Tokamak à Configuration Variable (TCV)**, a magnetic confinement fusion device operating at **Swiss Plasma Center (SPC)**.

## Prerequisites

Ensure you have the following dependencies installed:

- **MATLAB** (with Deep Learning Toolbox)
- **Python** (tested with 3.10.8)
- **PyTorch** (tested with 1.13.0+cpu)
- **NumPy** (tested with 1.23.4)

## Setup

### Python Virtual Environment Setup

Run the following commands in a Windows terminal:

```sh
python -m venv env
env\Scripts\activate
pip install torch==1.13.0+cpu torchvision==0.14.0+cpu torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cpu
pip install numpy==1.23.4
```

Verify installation:

```sh
python -m pip show numpy torch torchvision
```

### MATLAB-Python Integration

Inside MATLAB, configure Python:

```matlab
pe = pyenv(ExecutionMode="OutOfProcess", Version="env\Scripts\python.exe");
pyenv
```

## Model Scripting

Convert and save a PyTorch model:

```python
scripted_model = torch.jit.script(model)
scripted_model.save('processed_model.pt')
print("Scripted model saved as 'processed_model.pt'")
```

## Data Preprocessing

### Load and Process Parquet Data in MATLAB

```matlab
Vars = {'FIR_core', 'PD', 'DML', 'IP'};
argsT = {'SelectedVariableNames', Vars};
PQ = parquetread("path_to_parquet_file", argsT{:});
PQ_array = table2array(PQ);
num_timesteps = size(PQ_array, 1);
max_timesteps = floor(num_timesteps / 40) * 40;
truncated_data = PQ_array(1:max_timesteps, :);
```

### Sliding Window Generation

```matlab
buffer = 0;
stride = 10;
t_win = 40;
windows = ds_sliding_window_generator(truncated_data, buffer, stride, t_win);
data_matrix = cell2mat(windows);
num_windows = floor(size(data_matrix, 1) / t_win);
reshaped_data_temp = reshape(data_matrix, [t_win, 4, num_windows]);
save('processed_data.mat', 'reshaped_data_temp');
```

### Sliding Window Function

```matlab
function windows = ds_sliding_window_generator(data_matrix, buffer, stride, t_win)
    [n_timestamps, ~] = size(data_matrix);
    starts = (buffer + 1):stride:(n_timestamps - t_win + 1);
    windows = cell(length(starts), 1);
    for i = 1:length(starts)
        windows{i} = data_matrix(starts(i):(starts(i) + t_win - 1), :);
    end
end
```

## Simulink Integration

### Slice Function Block

```matlab
function slice = getSlice(data, step)
    slice = squeeze(data(:, :, step)); % Extracts a 4x40 slice
end
```

### Normalization Function Block

```matlab
function y = normalize(u)
    if isempty(u) || ~ismatrix(u)
        error('Input must be a non-empty matrix.');
    end
    mu = mean(u, 2, 'omitnan');
    stdev = std(u, 0, 2, 'omitnan');
    stdev(stdev == 0) = 1;
    y = (u - mu) ./ stdev;
end
```

## Common Errors & Fixes

### 1. ImportError: `ModuleDeprecationWarning` from `numpy._globals`
Check installed packages:

```sh
pip list
```

### 2. NumPy Module Not Found
Ensure NumPy is in the system path:

```sh
echo $PATH
set PATH=%PATH%;C:\path\to\numpy
```

### 3. ImportError: `_imaging` from `PIL`
Reinstall Pillow:

```sh
pip uninstall PIL
pip uninstall pillow
pip install pillow
```

### 4. `torchvision` Object Not Callable in MATLAB

```matlab
torchvision = py.importlib.import_module('torchvision');
models = py.importlib.import_module('torchvision.models');
```

## References

- [MATLAB Deep Learning Toolbox](https://www.mathworks.com/products/deep-learning.html)
- [PyTorch Official Site](https://pytorch.org/get-started/previous-versions/)
- [NumPy Documentation](https://numpy.org/)
- [MATLAB-PyTorch Integration Example](https://github.com/matlab-deep-learning/compare-PyTorch-models-from-MATLAB)

---

This repository is actively maintained. Feel free to raise issues or contribute! 🚀

