% Load the Parquet file data
Vars = {'FIR_core', 'PD', 'DML', 'IP'};
argsT = {'SelectedVariableNames', Vars};
PQ = parquetread("C:\Users\suhan\defuse\DEFUSE_tests\data_tests\source_parquet\TCV_DATAno79827.parquet", argsT{:});
PQ_array = table2array(PQ);  % Convert to array

% Ensure the number of timesteps is divisible by the window size (40 timesteps)
num_timesteps = size(PQ_array, 1);
max_timesteps = floor(num_timesteps / 40) * 40;  % Adjust truncation
truncated_data = PQ_array(1:max_timesteps, :);  % Truncate the data

% Sliding Window Generation
buffer = 0;  
stride = 10;  
t_win = 40; 
windows = ds_sliding_window_generator(truncated_data, buffer, stride, t_win);
data_matrix = cell2mat(windows);

[num_rows, num_features] = size(data_matrix);
truncated_rows = floor(num_rows / t_win) * t_win; % Ensure the number of rows is divisible by t_win (40 timesteps)
truncated_data_matrix = data_matrix(1:truncated_rows, :);
num_windows = truncated_rows / t_win;

reshaped_data_temp = reshape(truncated_data_matrix, [t_win, 4, num_windows]); % reshape into [timesteps x features x windows] 
% reshaped_data = permute(reshaped_data_temp, [2, 1, 3]); % permute the dimensions[features x timesteps x windows]
% save('processed_data.mat', 'reshaped_data');

% Sliding window generator function
function windows = ds_sliding_window_generator(data_matrix, buffer, stride, t_win)
    [n_timestamps, ~] = size(data_matrix);
    windows = {};  % Initialize cell array to store windows

    valid_start = buffer + 1;
    valid_end = n_timestamps - t_win + 1;
    starts = valid_start:stride:valid_end;

    n_windows = length(starts);
    windows = cell(n_windows, 1);  % Preallocate for windows

    for i = 1:n_windows
        window_data = data_matrix(starts(i):(starts(i) + t_win - 1), :);
        windows{i} = window_data;
    end
end
