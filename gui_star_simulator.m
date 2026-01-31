function gui_star_simulator()
% GUI_STAR_SIMULATOR Star Tracker 시뮬레이션 인터랙티브 GUI
%
% 사용법:
%   gui_star_simulator
%
% 기능:
%   - 센서/노출/관측 파라미터를 슬라이더/프리셋으로 조절
%   - 이상적 Grayscale vs 변환된 Grayscale 이미지 비교
%   - FOV, SNR, 별 검출률, Centroid 정확도 등 실시간 메트릭
%   - 5가지 Bayer→Gray 변환 방법 비교 (optimal 포함)
%
% 의존성:
%   core/simulate_star_image_realistic.m
%   core/bayer_to_gray_direct.m
%   utils/detect_stars_simple.m
%   utils/calculate_peak_snr.m
%   utils/evaluate_centroid_accuracy.m

%% 경로 설정
script_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(script_dir, 'core'));
addpath(fullfile(script_dir, 'utils'));

%% 카탈로그 사전 로드
catalog = preload_catalog(script_dir);

%% 상태 초기화
app = struct();
app.catalog = catalog;
app.params = init_params();
app.cache = struct('gray_ideal', [], 'bayer_img', [], 'star_info', [], ...
    'gray_converted', [], 'method_info', [], ...
    'detection', [], 'snr_db', 0, 'centroid_rms', 0);
app.dirty = struct('stage1', true, 'stage2', true, 'stage3', true);

%% 메인 윈도우
app.fig = uifigure('Name', 'Star Tracker Simulator', ...
    'Position', [50 50 1600 900], 'Color', [0.15 0.15 0.18], ...
    'Resize', 'off');

%% 좌측 패널: 탭
left_panel = uipanel(app.fig, 'Position', [0 0 400 900], ...
    'BackgroundColor', [0.18 0.18 0.22], 'BorderType', 'none');
app.tabgroup = uitabgroup(left_panel, 'Position', [5 5 390 890]);

%% 탭 생성
app = create_sensor_tab(app);
app = create_exposure_tab(app);
app = create_noise_tab(app);
app = create_observation_tab(app);
app = create_processing_tab(app);

%% 우측: 버튼, 이미지, 메트릭
app = create_action_buttons(app);
app = create_image_area(app);
app = create_metrics_panel(app);

%% 초기 FOV/노이즈 계산
update_fov_display(app);
update_noise_budget(app);

%% 저장
app.fig.UserData = app;
end

%% ========== 카탈로그 사전 로드 ==========
function catalog = preload_catalog(script_dir)
    data_dir = fullfile(script_dir, 'data');
    mat_path = fullfile(data_dir, 'star_catalog_kvector.mat');
    csv_path = fullfile(data_dir, 'Hipparcos_Below_6.0.csv');

    catalog = [];
    if exist(mat_path, 'file')
        fprintf('카탈로그 로드 중 (star_catalog_kvector.mat)...\n');
        data = load(mat_path);
        catalog = data.catalog_data;
        fprintf('카탈로그 로드 완료 (%d stars)\n', length(catalog.star_catalog.RA));
    elseif exist(csv_path, 'file')
        fprintf('카탈로그 로드 중 (CSV)...\n');
        opts = detectImportOptions(csv_path);
        tbl = readtable(csv_path, opts);
        catalog.star_catalog.RA = tbl.RA;
        catalog.star_catalog.DEC = tbl.DE;
        catalog.star_catalog.Magnitude = tbl.Magnitude;
        fprintf('CSV 카탈로그 로드 완료\n');
    else
        warning('별 카탈로그를 찾을 수 없습니다. data/ 폴더를 확인하세요.');
    end
end

%% ========== 기본 파라미터 ==========
function p = init_params()
    % Sensor
    p.pixel_size = 2.0;         % [um]
    p.focal_length = 10.42;     % [mm]
    p.res_w = 1280;             % [px]
    p.res_h = 720;              % [px]
    p.qe = 0.5;                 % [0~1]
    p.mag_limit = 6.5;

    % Exposure/Gain
    p.exposure_ms = 22.0;       % [ms]
    p.analog_gain = 16.0;       % [x]
    p.digital_gain = 1.0;       % [x]

    % Noise
    p.noise_enabled = true;
    p.dark_current = 0.1;       % [e-/px/s]
    p.read_noise = 3.0;         % [e- RMS]

    % Bayer channel sensitivity
    p.sens_R = 1.0;
    p.sens_G = 1.0;
    p.sens_B = 0.9;

    % Observation
    p.ra = 84.0;                % [deg]
    p.dec = -1.0;               % [deg]
    p.roll = 0.0;               % [deg]

    % Processing
    p.conv_method = 'raw';
    p.opt_weights = [0.4544, 0.3345, 0.2111];
    p.threshold = 15;           % [ADU]
    p.min_area = 2;             % [px]

    % Display
    p.show_overlay = false;
    p.show_labels = false;
end

%% ========== Tab 1: Sensor ==========
function app = create_sensor_tab(app)
    tab = uitab(app.tabgroup, 'Title', 'Sensor', ...
        'BackgroundColor', [0.2 0.2 0.24]);

    y = 820;  % 시작 y좌표 (위에서 아래로)

    % --- Pixel Size ---
    uilabel(tab, 'Position', [10 y 380 20], 'Text', 'Pixel Size [um]', ...
        'FontColor', 'w', 'FontWeight', 'bold');
    y = y - 30;
    app.h.slider_pix = uislider(tab, 'Position', [10 y 260 3], ...
        'Limits', [1 10], 'Value', app.params.pixel_size, 'MajorTicks', 1:10);
    app.h.edit_pix = uieditfield(tab, 'numeric', 'Position', [290 y-10 70 22], ...
        'Value', app.params.pixel_size, 'Limits', [1 10], ...
        'ValueDisplayFormat', '%.1f');
    app.h.slider_pix.ValueChangedFcn = @(s,e) cb_sensor(s, e, 'pixel_size', app.fig);
    app.h.edit_pix.ValueChangedFcn = @(s,e) cb_sensor_edit(s, e, 'pixel_size', app.fig);

    % --- Focal Length ---
    y = y - 50;
    uilabel(tab, 'Position', [10 y 380 20], 'Text', 'Focal Length [mm]', ...
        'FontColor', 'w', 'FontWeight', 'bold');
    y = y - 30;
    app.h.slider_fl = uislider(tab, 'Position', [10 y 260 3], ...
        'Limits', [5 50], 'Value', app.params.focal_length, 'MajorTicks', [5 10 20 30 40 50]);
    app.h.edit_fl = uieditfield(tab, 'numeric', 'Position', [290 y-10 70 22], ...
        'Value', app.params.focal_length, 'Limits', [5 50], ...
        'ValueDisplayFormat', '%.2f');
    app.h.slider_fl.ValueChangedFcn = @(s,e) cb_sensor(s, e, 'focal_length', app.fig);
    app.h.edit_fl.ValueChangedFcn = @(s,e) cb_sensor_edit(s, e, 'focal_length', app.fig);

    % --- Resolution Presets ---
    y = y - 55;
    uilabel(tab, 'Position', [10 y 380 20], 'Text', 'Resolution', ...
        'FontColor', 'w', 'FontWeight', 'bold');
    y = y - 30;
    uibutton(tab, 'Position', [10 y 110 28], 'Text', '640x360', ...
        'ButtonPushedFcn', @(s,e) cb_res_preset(s, e, 640, 360, app.fig));
    uibutton(tab, 'Position', [130 y 120 28], 'Text', '1280x720', ...
        'ButtonPushedFcn', @(s,e) cb_res_preset(s, e, 1280, 720, app.fig));
    uibutton(tab, 'Position', [260 y 110 28], 'Text', '2688x1520', ...
        'ButtonPushedFcn', @(s,e) cb_res_preset(s, e, 2688, 1520, app.fig));

    y = y - 35;
    uilabel(tab, 'Position', [10 y 30 22], 'Text', 'W:', 'FontColor', [0.7 0.7 0.7]);
    app.h.edit_resw = uieditfield(tab, 'numeric', 'Position', [35 y 70 22], ...
        'Value', app.params.res_w, 'Limits', [100 4000], 'RoundFractionalValues', 'on');
    uilabel(tab, 'Position', [120 y 30 22], 'Text', 'H:', 'FontColor', [0.7 0.7 0.7]);
    app.h.edit_resh = uieditfield(tab, 'numeric', 'Position', [145 y 70 22], ...
        'Value', app.params.res_h, 'Limits', [100 3000], 'RoundFractionalValues', 'on');
    app.h.edit_resw.ValueChangedFcn = @(s,e) cb_res_changed(s, e, app.fig);
    app.h.edit_resh.ValueChangedFcn = @(s,e) cb_res_changed(s, e, app.fig);

    % --- Quantum Efficiency ---
    y = y - 50;
    uilabel(tab, 'Position', [10 y 380 20], 'Text', 'Quantum Efficiency', ...
        'FontColor', 'w', 'FontWeight', 'bold');
    y = y - 30;
    app.h.slider_qe = uislider(tab, 'Position', [10 y 260 3], ...
        'Limits', [0.1 0.9], 'Value', app.params.qe, 'MajorTicks', 0.1:0.1:0.9);
    app.h.edit_qe = uieditfield(tab, 'numeric', 'Position', [290 y-10 70 22], ...
        'Value', app.params.qe, 'Limits', [0.1 0.9], ...
        'ValueDisplayFormat', '%.2f');
    app.h.slider_qe.ValueChangedFcn = @(s,e) cb_sensor(s, e, 'qe', app.fig);
    app.h.edit_qe.ValueChangedFcn = @(s,e) cb_sensor_edit(s, e, 'qe', app.fig);

    % --- Magnitude Limit ---
    y = y - 55;
    uilabel(tab, 'Position', [10 y 380 20], 'Text', 'Star Magnitude Limit', ...
        'FontColor', 'w', 'FontWeight', 'bold');
    y = y - 30;
    app.h.slider_mag = uislider(tab, 'Position', [10 y 260 3], ...
        'Limits', [3 8], 'Value', app.params.mag_limit, 'MajorTicks', 3:8);
    app.h.edit_mag = uieditfield(tab, 'numeric', 'Position', [290 y-10 70 22], ...
        'Value', app.params.mag_limit, 'Limits', [3 8], ...
        'ValueDisplayFormat', '%.1f');
    app.h.slider_mag.ValueChangedFcn = @(s,e) cb_sensor(s, e, 'mag_limit', app.fig);
    app.h.edit_mag.ValueChangedFcn = @(s,e) cb_sensor_edit(s, e, 'mag_limit', app.fig);

    % --- FOV Display ---
    y = y - 60;
    fov_panel = uipanel(tab, 'Position', [10 y-80 360 100], ...
        'Title', 'Calculated FOV', 'FontWeight', 'bold', ...
        'BackgroundColor', [0.15 0.2 0.15], 'ForegroundColor', [0.5 1 0.5]);
    app.h.lbl_fov_h = uilabel(fov_panel, 'Position', [10 55 340 20], ...
        'Text', 'H: --', 'FontColor', [0.5 1 0.5], 'FontName', 'Consolas');
    app.h.lbl_fov_v = uilabel(fov_panel, 'Position', [10 35 340 20], ...
        'Text', 'V: --', 'FontColor', [0.5 1 0.5], 'FontName', 'Consolas');
    app.h.lbl_fov_d = uilabel(fov_panel, 'Position', [10 15 340 20], ...
        'Text', 'Diagonal: --', 'FontColor', [0.5 1 0.5], 'FontName', 'Consolas');
    app.h.lbl_ifov = uilabel(fov_panel, 'Position', [200 55 160 20], ...
        'Text', 'IFOV: --', 'FontColor', [0.5 1 0.5], 'FontName', 'Consolas');
end

%% ========== Tab 2: Exposure/Gain ==========
function app = create_exposure_tab(app)
    tab = uitab(app.tabgroup, 'Title', 'Exposure', ...
        'BackgroundColor', [0.2 0.2 0.24]);

    y = 820;

    % --- Exposure Time ---
    uilabel(tab, 'Position', [10 y 380 20], 'Text', 'Exposure Time [ms]', ...
        'FontColor', 'w', 'FontWeight', 'bold');
    y = y - 25;
    app.h.lbl_exp = uilabel(tab, 'Position', [10 y 380 25], ...
        'Text', '22.0 ms', 'FontColor', [1 0.9 0.4], ...
        'FontName', 'Consolas', 'FontSize', 16, 'FontWeight', 'bold');
    y = y - 35;

    % 로그 스케일 슬라이더 (log2(1)=0 ~ log2(500)=~8.97)
    app.h.slider_exp = uislider(tab, 'Position', [10 y 360 3], ...
        'Limits', [0 log2(500)], 'Value', log2(app.params.exposure_ms));
    app.h.slider_exp.ValueChangedFcn = @(s,e) cb_exp_slider(s, e, app.fig);

    y = y - 30;
    uibutton(tab, 'Position', [10 y 55 26], 'Text', '1ms', ...
        'ButtonPushedFcn', @(s,e) cb_exp_preset(s, e, 1, app.fig));
    uibutton(tab, 'Position', [70 y 55 26], 'Text', '5ms', ...
        'ButtonPushedFcn', @(s,e) cb_exp_preset(s, e, 5, app.fig));
    uibutton(tab, 'Position', [130 y 55 26], 'Text', '22ms', ...
        'ButtonPushedFcn', @(s,e) cb_exp_preset(s, e, 22, app.fig));
    uibutton(tab, 'Position', [190 y 55 26], 'Text', '50ms', ...
        'ButtonPushedFcn', @(s,e) cb_exp_preset(s, e, 50, app.fig));
    uibutton(tab, 'Position', [250 y 55 26], 'Text', '100ms', ...
        'ButtonPushedFcn', @(s,e) cb_exp_preset(s, e, 100, app.fig));
    uibutton(tab, 'Position', [310 y 60 26], 'Text', '500ms', ...
        'ButtonPushedFcn', @(s,e) cb_exp_preset(s, e, 500, app.fig));

    % --- Analog Gain ---
    y = y - 50;
    uilabel(tab, 'Position', [10 y 380 20], 'Text', 'Analog Gain', ...
        'FontColor', 'w', 'FontWeight', 'bold');
    y = y - 25;
    app.h.lbl_gain = uilabel(tab, 'Position', [10 y 380 25], ...
        'Text', '16.0x', 'FontColor', [1 0.9 0.4], ...
        'FontName', 'Consolas', 'FontSize', 16, 'FontWeight', 'bold');
    y = y - 35;

    % 프리셋 버튼 (Python UI 스타일)
    gains = [1 2 4 8 16 32 64];
    for i = 1:length(gains)
        uibutton(tab, 'Position', [10+(i-1)*52 y 48 26], ...
            'Text', sprintf('%dx', gains(i)), ...
            'ButtonPushedFcn', @(s,e) cb_gain_preset(s, e, gains(i), app.fig));
    end

    y = y - 35;
    % 미세 조정 + 슬라이더
    uibutton(tab, 'Position', [10 y 45 26], 'Text', '-8', ...
        'ButtonPushedFcn', @(s,e) cb_gain_adjust(s, e, -8, app.fig));
    uibutton(tab, 'Position', [60 y 45 26], 'Text', '-1', ...
        'ButtonPushedFcn', @(s,e) cb_gain_adjust(s, e, -1, app.fig));
    app.h.slider_gain = uislider(tab, 'Position', [115 y 165 3], ...
        'Limits', [1 64], 'Value', app.params.analog_gain, ...
        'MajorTicks', [1 4 8 16 32 64]);
    app.h.slider_gain.ValueChangedFcn = @(s,e) cb_gain_slider(s, e, app.fig);
    uibutton(tab, 'Position', [295 y-10 45 26], 'Text', '+1', ...
        'ButtonPushedFcn', @(s,e) cb_gain_adjust(s, e, 1, app.fig));
    uibutton(tab, 'Position', [345 y-10 45 26], 'Text', '+8', ...
        'ButtonPushedFcn', @(s,e) cb_gain_adjust(s, e, 8, app.fig));

    % --- Digital Gain ---
    y = y - 60;
    uilabel(tab, 'Position', [10 y 380 20], 'Text', 'Digital Gain [x]', ...
        'FontColor', 'w', 'FontWeight', 'bold');
    y = y - 30;
    app.h.slider_dgain = uislider(tab, 'Position', [10 y 260 3], ...
        'Limits', [0.5 4], 'Value', app.params.digital_gain, ...
        'MajorTicks', [0.5 1 2 3 4]);
    app.h.edit_dgain = uieditfield(tab, 'numeric', 'Position', [290 y-10 70 22], ...
        'Value', app.params.digital_gain, 'Limits', [0.5 4], ...
        'ValueDisplayFormat', '%.1f');
    app.h.slider_dgain.ValueChangedFcn = @(s,e) cb_dgain(s, e, app.fig);
    app.h.edit_dgain.ValueChangedFcn = @(s,e) cb_dgain_edit(s, e, app.fig);

    % --- Signal Estimate ---
    y = y - 70;
    sig_panel = uipanel(tab, 'Position', [10 y-90 360 110], ...
        'Title', 'Signal Estimate (peak ADU)', 'FontWeight', 'bold', ...
        'BackgroundColor', [0.15 0.15 0.2], 'ForegroundColor', [0.6 0.8 1]);
    app.h.lbl_sig1 = uilabel(sig_panel, 'Position', [10 65 340 20], ...
        'Text', '1st mag: --', 'FontColor', [0.6 0.8 1], 'FontName', 'Consolas');
    app.h.lbl_sig3 = uilabel(sig_panel, 'Position', [10 45 340 20], ...
        'Text', '3rd mag: --', 'FontColor', [0.6 0.8 1], 'FontName', 'Consolas');
    app.h.lbl_sig6 = uilabel(sig_panel, 'Position', [10 25 340 20], ...
        'Text', '6th mag: --', 'FontColor', [0.6 0.8 1], 'FontName', 'Consolas');
    app.h.lbl_sigsat = uilabel(sig_panel, 'Position', [10 5 340 20], ...
        'Text', 'Saturation: 255 ADU', 'FontColor', [1 0.5 0.5], 'FontName', 'Consolas');
end

%% ========== Tab 3: Noise ==========
function app = create_noise_tab(app)
    tab = uitab(app.tabgroup, 'Title', 'Noise', ...
        'BackgroundColor', [0.2 0.2 0.24]);

    y = 820;

    % --- Noise Enable ---
    app.h.chk_noise = uicheckbox(tab, 'Position', [10 y 200 22], ...
        'Text', 'Enable Noise', 'Value', app.params.noise_enabled, ...
        'FontColor', 'w', 'FontSize', 14, 'FontWeight', 'bold');
    app.h.chk_noise.ValueChangedFcn = @(s,e) cb_noise_toggle(s, e, app.fig);

    % --- Dark Current ---
    y = y - 50;
    uilabel(tab, 'Position', [10 y 380 20], 'Text', 'Dark Current [e-/px/s]', ...
        'FontColor', 'w', 'FontWeight', 'bold');
    y = y - 30;
    app.h.slider_dark = uislider(tab, 'Position', [10 y 260 3], ...
        'Limits', [0 1], 'Value', app.params.dark_current, ...
        'MajorTicks', 0:0.2:1);
    app.h.edit_dark = uieditfield(tab, 'numeric', 'Position', [290 y-10 70 22], ...
        'Value', app.params.dark_current, 'Limits', [0 1], ...
        'ValueDisplayFormat', '%.3f');
    app.h.slider_dark.ValueChangedFcn = @(s,e) cb_noise_param(s, e, 'dark_current', app.fig);
    app.h.edit_dark.ValueChangedFcn = @(s,e) cb_noise_param_edit(s, e, 'dark_current', app.fig);

    y = y - 30;
    uibutton(tab, 'Position', [10 y 100 26], 'Text', 'Space (0.01)', ...
        'ButtonPushedFcn', @(s,e) cb_dark_preset(s, e, 0.01, app.fig));
    uibutton(tab, 'Position', [115 y 100 26], 'Text', 'Ground (0.1)', ...
        'ButtonPushedFcn', @(s,e) cb_dark_preset(s, e, 0.1, app.fig));

    % --- Read Noise ---
    y = y - 50;
    uilabel(tab, 'Position', [10 y 380 20], 'Text', 'Read Noise [e- RMS]', ...
        'FontColor', 'w', 'FontWeight', 'bold');
    y = y - 30;
    app.h.slider_read = uislider(tab, 'Position', [10 y 260 3], ...
        'Limits', [0 20], 'Value', app.params.read_noise, ...
        'MajorTicks', 0:5:20);
    app.h.edit_read = uieditfield(tab, 'numeric', 'Position', [290 y-10 70 22], ...
        'Value', app.params.read_noise, 'Limits', [0 20], ...
        'ValueDisplayFormat', '%.1f');
    app.h.slider_read.ValueChangedFcn = @(s,e) cb_noise_param(s, e, 'read_noise', app.fig);
    app.h.edit_read.ValueChangedFcn = @(s,e) cb_noise_param_edit(s, e, 'read_noise', app.fig);

    % --- Noise Budget ---
    y = y - 70;
    noise_panel = uipanel(tab, 'Position', [10 y-90 360 110], ...
        'Title', 'Noise Budget', 'FontWeight', 'bold', ...
        'BackgroundColor', [0.2 0.15 0.15], 'ForegroundColor', [1 0.7 0.7]);
    app.h.lbl_nshot = uilabel(noise_panel, 'Position', [10 65 340 20], ...
        'Text', 'Shot noise (6mag): --', 'FontColor', [1 0.7 0.7], 'FontName', 'Consolas');
    app.h.lbl_nread = uilabel(noise_panel, 'Position', [10 45 340 20], ...
        'Text', 'Read noise (ADU): --', 'FontColor', [1 0.7 0.7], 'FontName', 'Consolas');
    app.h.lbl_ndark = uilabel(noise_panel, 'Position', [10 25 340 20], ...
        'Text', 'Dark current (ADU): --', 'FontColor', [1 0.7 0.7], 'FontName', 'Consolas');
    app.h.lbl_ntotal = uilabel(noise_panel, 'Position', [10 5 340 20], ...
        'Text', 'Total noise RMS: --', 'FontColor', [1 0.5 0.5], ...
        'FontName', 'Consolas', 'FontWeight', 'bold');

    % --- Scene Presets ---
    y = y - 130;
    uilabel(tab, 'Position', [10 y 380 20], 'Text', 'Scene Presets', ...
        'FontColor', 'w', 'FontWeight', 'bold');
    y = y - 30;
    uibutton(tab, 'Position', [10 y 85 28], 'Text', 'Space', ...
        'BackgroundColor', [0.2 0.3 0.5], ...
        'ButtonPushedFcn', @(s,e) cb_scene_preset(s, e, 'space', app.fig));
    uibutton(tab, 'Position', [100 y 85 28], 'Text', 'Ground', ...
        'BackgroundColor', [0.3 0.3 0.2], ...
        'ButtonPushedFcn', @(s,e) cb_scene_preset(s, e, 'ground', app.fig));
    uibutton(tab, 'Position', [190 y 85 28], 'Text', 'Low Noise', ...
        'BackgroundColor', [0.2 0.35 0.2], ...
        'ButtonPushedFcn', @(s,e) cb_scene_preset(s, e, 'low_noise', app.fig));
    uibutton(tab, 'Position', [280 y 85 28], 'Text', 'High Noise', ...
        'BackgroundColor', [0.4 0.2 0.2], ...
        'ButtonPushedFcn', @(s,e) cb_scene_preset(s, e, 'high_noise', app.fig));
end

%% ========== Tab 4: Observation ==========
function app = create_observation_tab(app)
    tab = uitab(app.tabgroup, 'Title', 'Observation', ...
        'BackgroundColor', [0.2 0.2 0.24]);

    y = 820;

    % --- RA ---
    uilabel(tab, 'Position', [10 y 380 20], 'Text', 'Right Ascension (RA) [deg]', ...
        'FontColor', 'w', 'FontWeight', 'bold');
    y = y - 30;
    app.h.slider_ra = uislider(tab, 'Position', [10 y 260 3], ...
        'Limits', [0 360], 'Value', app.params.ra, 'MajorTicks', 0:60:360);
    app.h.edit_ra = uieditfield(tab, 'numeric', 'Position', [290 y-10 70 22], ...
        'Value', app.params.ra, 'Limits', [0 360], ...
        'ValueDisplayFormat', '%.2f');
    app.h.slider_ra.ValueChangedFcn = @(s,e) cb_obs(s, e, 'ra', app.fig);
    app.h.edit_ra.ValueChangedFcn = @(s,e) cb_obs_edit(s, e, 'ra', app.fig);

    % --- DEC ---
    y = y - 55;
    uilabel(tab, 'Position', [10 y 380 20], 'Text', 'Declination (DEC) [deg]', ...
        'FontColor', 'w', 'FontWeight', 'bold');
    y = y - 30;
    app.h.slider_dec = uislider(tab, 'Position', [10 y 260 3], ...
        'Limits', [-90 90], 'Value', app.params.dec, 'MajorTicks', -90:30:90);
    app.h.edit_dec = uieditfield(tab, 'numeric', 'Position', [290 y-10 70 22], ...
        'Value', app.params.dec, 'Limits', [-90 90], ...
        'ValueDisplayFormat', '%.2f');
    app.h.slider_dec.ValueChangedFcn = @(s,e) cb_obs(s, e, 'dec', app.fig);
    app.h.edit_dec.ValueChangedFcn = @(s,e) cb_obs_edit(s, e, 'dec', app.fig);

    % --- Roll ---
    y = y - 55;
    uilabel(tab, 'Position', [10 y 380 20], 'Text', 'Roll [deg]', ...
        'FontColor', 'w', 'FontWeight', 'bold');
    y = y - 30;
    app.h.slider_roll = uislider(tab, 'Position', [10 y 260 3], ...
        'Limits', [0 360], 'Value', app.params.roll, 'MajorTicks', 0:45:360);
    app.h.edit_roll = uieditfield(tab, 'numeric', 'Position', [290 y-10 70 22], ...
        'Value', app.params.roll, 'Limits', [0 360], ...
        'ValueDisplayFormat', '%.1f');
    app.h.slider_roll.ValueChangedFcn = @(s,e) cb_obs(s, e, 'roll', app.fig);
    app.h.edit_roll.ValueChangedFcn = @(s,e) cb_obs_edit(s, e, 'roll', app.fig);

    % --- Observation Presets ---
    y = y - 60;
    uilabel(tab, 'Position', [10 y 380 20], 'Text', 'Target Presets', ...
        'FontColor', 'w', 'FontWeight', 'bold');
    y = y - 35;
    presets = {
        'Orion Belt', 84, -1;
        'Polaris', 37.95, 89.26;
        'Big Dipper', 165, 55;
        'Antares', 247.35, -26.43;
        'Random', -1, -1;
    };
    for i = 1:size(presets, 1)
        uibutton(tab, 'Position', [10 y-(i-1)*32 360 28], ...
            'Text', sprintf('%s  (RA=%.1f, DEC=%.1f)', presets{i,1}, presets{i,2}, presets{i,3}), ...
            'ButtonPushedFcn', @(s,e) cb_obs_preset(s, e, presets{i,2}, presets{i,3}, app.fig));
    end
end

%% ========== Tab 5: Processing ==========
function app = create_processing_tab(app)
    tab = uitab(app.tabgroup, 'Title', 'Processing', ...
        'BackgroundColor', [0.2 0.2 0.24]);

    y = 820;

    % --- Conversion Method ---
    uilabel(tab, 'Position', [10 y 380 20], 'Text', 'Bayer -> Gray Method', ...
        'FontColor', 'w', 'FontWeight', 'bold');
    y = y - 30;
    app.h.dd_method = uidropdown(tab, 'Position', [10 y 360 28], ...
        'Items', {'raw', 'binning', 'green', 'weighted', 'optimal'}, ...
        'Value', app.params.conv_method);
    app.h.dd_method.ValueChangedFcn = @(s,e) cb_method_changed(s, e, app.fig);

    % --- Optimal Weights ---
    y = y - 45;
    app.h.panel_weights = uipanel(tab, 'Position', [10 y-130 360 160], ...
        'Title', 'Optimal Weights (SNR Maximized)', 'FontWeight', 'bold', ...
        'BackgroundColor', [0.2 0.18 0.22], 'ForegroundColor', [1 0.8 1]);

    wp = app.h.panel_weights;
    uilabel(wp, 'Position', [10 115 30 20], 'Text', 'R:', ...
        'FontColor', [1 0.4 0.4], 'FontWeight', 'bold');
    app.h.slider_wR = uislider(wp, 'Position', [35 115 220 3], ...
        'Limits', [0 1], 'Value', app.params.opt_weights(1));
    app.h.lbl_wR = uilabel(wp, 'Position', [270 105 80 20], ...
        'Text', sprintf('%.4f', app.params.opt_weights(1)), ...
        'FontColor', [1 0.4 0.4], 'FontName', 'Consolas');

    uilabel(wp, 'Position', [10 80 30 20], 'Text', 'G:', ...
        'FontColor', [0.4 1 0.4], 'FontWeight', 'bold');
    app.h.slider_wG = uislider(wp, 'Position', [35 80 220 3], ...
        'Limits', [0 1], 'Value', app.params.opt_weights(2));
    app.h.lbl_wG = uilabel(wp, 'Position', [270 70 80 20], ...
        'Text', sprintf('%.4f', app.params.opt_weights(2)), ...
        'FontColor', [0.4 1 0.4], 'FontName', 'Consolas');

    uilabel(wp, 'Position', [10 45 30 20], 'Text', 'B:', ...
        'FontColor', [0.4 0.6 1], 'FontWeight', 'bold');
    app.h.slider_wB = uislider(wp, 'Position', [35 45 220 3], ...
        'Limits', [0 1], 'Value', app.params.opt_weights(3));
    app.h.lbl_wB = uilabel(wp, 'Position', [270 35 80 20], ...
        'Text', sprintf('%.4f', app.params.opt_weights(3)), ...
        'FontColor', [0.4 0.6 1], 'FontName', 'Consolas');

    app.h.lbl_wsum = uilabel(wp, 'Position', [10 10 100 20], ...
        'Text', 'Sum: 1.000', 'FontColor', 'w', 'FontName', 'Consolas');
    uibutton(wp, 'Position', [130 5 100 22], 'Text', 'Normalize', ...
        'ButtonPushedFcn', @(s,e) cb_normalize_weights(s, e, app.fig));
    uibutton(wp, 'Position', [240 5 100 22], 'Text', 'Reset', ...
        'ButtonPushedFcn', @(s,e) cb_reset_weights(s, e, app.fig));

    app.h.slider_wR.ValueChangedFcn = @(s,e) cb_weight_changed(s, e, app.fig);
    app.h.slider_wG.ValueChangedFcn = @(s,e) cb_weight_changed(s, e, app.fig);
    app.h.slider_wB.ValueChangedFcn = @(s,e) cb_weight_changed(s, e, app.fig);

    % optimal이 아닌 경우 패널 비활성화 표시
    app.h.panel_weights.Visible = strcmp(app.params.conv_method, 'optimal');

    % --- Star Detection ---
    y = y - 195;
    uilabel(tab, 'Position', [10 y 380 20], 'Text', 'Star Detection', ...
        'FontColor', 'w', 'FontWeight', 'bold');

    y = y - 25;
    uilabel(tab, 'Position', [10 y 150 20], 'Text', 'Threshold [ADU]:', ...
        'FontColor', [0.7 0.7 0.7]);
    app.h.slider_thresh = uislider(tab, 'Position', [10 y-25 260 3], ...
        'Limits', [1 100], 'Value', app.params.threshold, ...
        'MajorTicks', [1 15 30 50 75 100]);
    app.h.edit_thresh = uieditfield(tab, 'numeric', 'Position', [290 y-35 70 22], ...
        'Value', app.params.threshold, 'Limits', [1 100], ...
        'RoundFractionalValues', 'on');
    app.h.slider_thresh.ValueChangedFcn = @(s,e) cb_detect_param(s, e, 'threshold', app.fig);
    app.h.edit_thresh.ValueChangedFcn = @(s,e) cb_detect_param_edit(s, e, 'threshold', app.fig);

    y = y - 65;
    uilabel(tab, 'Position', [10 y 150 20], 'Text', 'Min Area [px]:', ...
        'FontColor', [0.7 0.7 0.7]);
    app.h.slider_minarea = uislider(tab, 'Position', [10 y-25 260 3], ...
        'Limits', [1 20], 'Value', app.params.min_area, ...
        'MajorTicks', [1 2 5 10 15 20]);
    app.h.edit_minarea = uieditfield(tab, 'numeric', 'Position', [290 y-35 70 22], ...
        'Value', app.params.min_area, 'Limits', [1 20], ...
        'RoundFractionalValues', 'on');
    app.h.slider_minarea.ValueChangedFcn = @(s,e) cb_detect_param(s, e, 'min_area', app.fig);
    app.h.edit_minarea.ValueChangedFcn = @(s,e) cb_detect_param_edit(s, e, 'min_area', app.fig);

    % --- Display Options ---
    y = y - 65;
    uilabel(tab, 'Position', [10 y 380 20], 'Text', 'Display Options', ...
        'FontColor', 'w', 'FontWeight', 'bold');
    y = y - 28;
    app.h.chk_overlay = uicheckbox(tab, 'Position', [10 y 200 22], ...
        'Text', 'Show Detection Overlay', 'Value', false, 'FontColor', [0.7 0.7 0.7]);
    app.h.chk_overlay.ValueChangedFcn = @(s,e) cb_display_opt(s, e, app.fig);
    y = y - 28;
    app.h.chk_labels = uicheckbox(tab, 'Position', [10 y 200 22], ...
        'Text', 'Show Magnitude Labels', 'Value', false, 'FontColor', [0.7 0.7 0.7]);
    app.h.chk_labels.ValueChangedFcn = @(s,e) cb_display_opt(s, e, app.fig);
end

%% ========== Action Buttons ==========
function app = create_action_buttons(app)
    btn_y = 862;

    app.h.btn_sim = uibutton(app.fig, 'Position', [415 btn_y 140 30], ...
        'Text', 'Simulate', 'FontSize', 14, 'FontWeight', 'bold', ...
        'BackgroundColor', [0.2 0.6 0.3], 'FontColor', 'w');
    app.h.btn_sim.ButtonPushedFcn = @(s,e) cb_simulate(s, e, app.fig);

    app.h.btn_quick = uibutton(app.fig, 'Position', [565 btn_y 140 30], ...
        'Text', 'Quick Update', 'FontSize', 12, ...
        'BackgroundColor', [0.2 0.4 0.6], 'FontColor', 'w');
    app.h.btn_quick.ButtonPushedFcn = @(s,e) cb_quick_update(s, e, app.fig);

    app.h.btn_export = uibutton(app.fig, 'Position', [715 btn_y 120 30], ...
        'Text', 'Export Images', 'FontSize', 11);
    app.h.btn_export.ButtonPushedFcn = @(s,e) cb_export(s, e, app.fig);

    % 상태 표시
    app.h.lbl_status = uilabel(app.fig, 'Position', [850 btn_y 500 30], ...
        'Text', 'Ready - Press [Simulate] to generate star images', ...
        'FontColor', [0.6 0.8 1], 'FontSize', 12, 'FontName', 'Consolas');
end

%% ========== Image Area ==========
function app = create_image_area(app)
    img_y = 225;
    img_h = 620;
    img_w = 565;

    % 좌: Ideal Grayscale
    uilabel(app.fig, 'Position', [415 img_y+img_h 560 22], ...
        'Text', 'Ideal Grayscale (No Noise)', ...
        'FontColor', [0.5 1 0.5], 'FontSize', 12, 'FontWeight', 'bold');
    app.h.ax_ideal = uiaxes(app.fig, 'Position', [415 img_y img_w img_h]);
    app.h.ax_ideal.Color = [0.05 0.05 0.08];
    app.h.ax_ideal.XTick = [];
    app.h.ax_ideal.YTick = [];
    app.h.ax_ideal.Box = 'on';
    app.h.ax_ideal.XColor = [0.3 0.3 0.3];
    app.h.ax_ideal.YColor = [0.3 0.3 0.3];
    title(app.h.ax_ideal, 'No simulation yet', 'Color', [0.5 0.5 0.5]);

    % 우: Converted Grayscale
    uilabel(app.fig, 'Position', [990 img_y+img_h 560 22], ...
        'Text', 'Converted Grayscale (Bayer -> Gray)', ...
        'FontColor', [1 0.8 0.4], 'FontSize', 12, 'FontWeight', 'bold');
    app.h.ax_conv = uiaxes(app.fig, 'Position', [990 img_y img_w img_h]);
    app.h.ax_conv.Color = [0.05 0.05 0.08];
    app.h.ax_conv.XTick = [];
    app.h.ax_conv.YTick = [];
    app.h.ax_conv.Box = 'on';
    app.h.ax_conv.XColor = [0.3 0.3 0.3];
    app.h.ax_conv.YColor = [0.3 0.3 0.3];
    title(app.h.ax_conv, 'No simulation yet', 'Color', [0.5 0.5 0.5]);
end

%% ========== Metrics Panel ==========
function app = create_metrics_panel(app)
    mp = uipanel(app.fig, 'Position', [415 5 1170 210], ...
        'BackgroundColor', [0.12 0.12 0.16], 'BorderType', 'line', ...
        'HighlightColor', [0.3 0.3 0.4]);

    % 좌측 열: FOV + 센서 정보
    col1 = 15;
    app.h.lbl_m_fov = uilabel(mp, 'Position', [col1 170 250 20], ...
        'Text', 'FOV: -- x -- deg', 'FontColor', [0.5 1 0.5], ...
        'FontName', 'Consolas', 'FontSize', 12);
    app.h.lbl_m_res = uilabel(mp, 'Position', [col1 148 250 20], ...
        'Text', 'Resolution: --', 'FontColor', [0.7 0.7 0.7], ...
        'FontName', 'Consolas', 'FontSize', 11);
    app.h.lbl_m_ifov = uilabel(mp, 'Position', [col1 126 250 20], ...
        'Text', 'IFOV: -- deg/px', 'FontColor', [0.7 0.7 0.7], ...
        'FontName', 'Consolas', 'FontSize', 11);
    app.h.lbl_m_pointing = uilabel(mp, 'Position', [col1 104 350 20], ...
        'Text', 'Pointing: RA=--, DEC=--', 'FontColor', [0.7 0.7 0.7], ...
        'FontName', 'Consolas', 'FontSize', 11);

    % 중앙 열: 별 정보
    col2 = 380;
    app.h.lbl_m_stars = uilabel(mp, 'Position', [col2 170 250 20], ...
        'Text', 'Stars in FOV: --', 'FontColor', [1 1 0.5], ...
        'FontName', 'Consolas', 'FontSize', 12, 'FontWeight', 'bold');
    app.h.lbl_m_detected = uilabel(mp, 'Position', [col2 148 250 20], ...
        'Text', 'Detected: -- (--)', 'FontColor', [0.7 0.7 0.7], ...
        'FontName', 'Consolas', 'FontSize', 11);
    app.h.lbl_m_snr = uilabel(mp, 'Position', [col2 126 250 20], ...
        'Text', 'Peak SNR: -- dB', 'FontColor', [0.7 0.7 0.7], ...
        'FontName', 'Consolas', 'FontSize', 11);
    app.h.lbl_m_rms = uilabel(mp, 'Position', [col2 104 250 20], ...
        'Text', 'Centroid RMS: -- px', 'FontColor', [0.7 0.7 0.7], ...
        'FontName', 'Consolas', 'FontSize', 11);

    % 우측 열: 처리 정보
    col3 = 730;
    app.h.lbl_m_method = uilabel(mp, 'Position', [col3 170 400 20], ...
        'Text', 'Method: --', 'FontColor', [1 0.8 0.4], ...
        'FontName', 'Consolas', 'FontSize', 12, 'FontWeight', 'bold');
    app.h.lbl_m_simtime = uilabel(mp, 'Position', [col3 148 300 20], ...
        'Text', 'Sim time: --', 'FontColor', [0.7 0.7 0.7], ...
        'FontName', 'Consolas', 'FontSize', 11);
    app.h.lbl_m_convtime = uilabel(mp, 'Position', [col3 126 300 20], ...
        'Text', 'Conv time: --', 'FontColor', [0.7 0.7 0.7], ...
        'FontName', 'Consolas', 'FontSize', 11);
    app.h.lbl_m_exposure = uilabel(mp, 'Position', [col3 104 300 20], ...
        'Text', 'Exp: -- | Gain: --', 'FontColor', [0.7 0.7 0.7], ...
        'FontName', 'Consolas', 'FontSize', 11);

    % 하단: 밝은 별 목록 (간략)
    app.h.lbl_m_brightest = uilabel(mp, 'Position', [col1 5 1100 90], ...
        'Text', 'Brightest stars: (run simulation first)', ...
        'FontColor', [0.5 0.5 0.6], 'FontName', 'Consolas', 'FontSize', 10, ...
        'VerticalAlignment', 'top');
end

%% ========== build_sensor_params: GUI -> struct ==========
function sp = build_sensor_params(app)
    p = app.params;
    sp = struct();
    sp.myu = p.pixel_size * 1e-6;        % um -> m
    sp.f = p.focal_length * 1e-3;         % mm -> m
    sp.l = p.res_w;
    sp.w = p.res_h;
    sp.mag_limit = p.mag_limit;
    sp.exposure_time = p.exposure_ms * 1e-3;  % ms -> s
    sp.analog_gain = p.analog_gain;
    sp.digital_gain = p.digital_gain;
    sp.quantum_efficiency = p.qe;
    sp.dark_current_rate = p.dark_current;
    sp.read_noise = p.read_noise;
    sp.add_noise = p.noise_enabled;
    sp.sensitivity_R = p.sens_R;
    sp.sensitivity_G = p.sens_G;
    sp.sensitivity_B = p.sens_B;

    % 카탈로그 사전 로드
    if ~isempty(app.catalog)
        sp.preloaded_catalog = app.catalog;
    end
end

%% ========== 시뮬레이션 파이프라인 ==========
function run_simulation(fig, force_stage1)
    app = fig.UserData;
    if nargin < 2, force_stage1 = true; end

    % Stage 1: Star image generation
    if app.dirty.stage1 || force_stage1
        app.h.lbl_status.Text = 'Simulating stars...';
        drawnow;

        sp = build_sensor_params(app);
        tic;
        [~, bayer_img, star_info] = ...
            simulate_star_image_realistic(app.params.ra, app.params.dec, ...
            app.params.roll, sp);
        sim_time = toc;

        % 노이즈 없는 이상적 이미지 사용 (star_info.ideal_gray)
        app.cache.gray_ideal = star_info.ideal_gray;
        app.cache.bayer_img = bayer_img;
        app.cache.star_info = star_info;
        app.cache.sim_time = sim_time;
        app.dirty.stage1 = false;
        app.dirty.stage2 = true;
        app.dirty.stage3 = true;
    end

    % Stage 2: Bayer -> Gray conversion
    if app.dirty.stage2
        app.h.lbl_status.Text = 'Converting Bayer -> Gray...';
        drawnow;

        method = app.params.conv_method;
        tic;
        if strcmp(method, 'optimal')
            [gray_conv, method_info] = bayer_to_gray_direct(...
                app.cache.bayer_img, method, app.params.opt_weights);
        else
            [gray_conv, method_info] = bayer_to_gray_direct(...
                app.cache.bayer_img, method);
        end
        conv_time = toc;

        app.cache.gray_converted = gray_conv;
        app.cache.method_info = method_info;
        app.cache.conv_time = conv_time;
        app.dirty.stage2 = false;
        app.dirty.stage3 = true;
    end

    % Stage 3: Detection + Metrics
    if app.dirty.stage3
        app.h.lbl_status.Text = 'Detecting stars...';
        drawnow;

        gray = app.cache.gray_converted;
        threshold = round(app.params.threshold);
        min_area = round(app.params.min_area);

        % 별 검출 (result struct 반환)
        det_result = detect_stars_simple(gray, threshold, min_area);
        app.cache.detection = det_result;

        % SNR
        app.cache.snr_db = calculate_peak_snr(gray);

        % Centroid accuracy
        if app.cache.star_info.num_stars > 0 && det_result.n_detected > 0
            true_cents = app.cache.star_info.true_centroids;
            % binning 보정
            if strcmp(app.params.conv_method, 'binning')
                true_cents = true_cents * 0.5;
            end
            cent_result = evaluate_centroid_accuracy(det_result, true_cents);
            app.cache.centroid_rms = cent_result.rms_error;
            app.cache.centroid_matched = cent_result.n_matched;
        else
            app.cache.centroid_rms = NaN;
            app.cache.centroid_matched = 0;
        end

        app.dirty.stage3 = false;
    end

    % 렌더링
    update_images(app);
    update_metrics(app);

    app.h.lbl_status.Text = sprintf('Done (sim: %.2fs, conv: %.3fs)', ...
        app.cache.sim_time, app.cache.conv_time);

    fig.UserData = app;
end

%% ========== 이미지 렌더링 ==========
function update_images(app)
    % 좌: Ideal
    ax1 = app.h.ax_ideal;
    cla(ax1);
    if ~isempty(app.cache.gray_ideal)
        ideal = uint8(min(255, app.cache.gray_ideal));
        imagesc(ax1, ideal);
        colormap(ax1, gray(256));
        axis(ax1, 'image');
        ax1.XTick = [];
        ax1.YTick = [];
        title(ax1, sprintf('Ideal (%dx%d)', size(ideal,2), size(ideal,1)), ...
            'Color', [0.5 1 0.5]);

        % 오버레이
        if app.params.show_overlay && app.cache.star_info.num_stars > 0
            hold(ax1, 'on');
            tc = app.cache.star_info.true_centroids;
            plot(ax1, tc(:,1), tc(:,2), 'go', 'MarkerSize', 10, 'LineWidth', 1.5);
            if app.params.show_labels
                mags = app.cache.star_info.magnitudes;
                [~, idx] = sort(mags);
                for i = 1:min(10, length(idx))
                    j = idx(i);
                    text(ax1, tc(j,1)+8, tc(j,2), sprintf('%.1f', mags(j)), ...
                        'Color', 'y', 'FontSize', 8);
                end
            end
            hold(ax1, 'off');
        end
    end

    % 우: Converted
    ax2 = app.h.ax_conv;
    cla(ax2);
    if ~isempty(app.cache.gray_converted)
        conv = app.cache.gray_converted;
        imagesc(ax2, conv);
        colormap(ax2, gray(256));
        axis(ax2, 'image');
        ax2.XTick = [];
        ax2.YTick = [];

        method_str = app.params.conv_method;
        title(ax2, sprintf('%s (%dx%d)', method_str, size(conv,2), size(conv,1)), ...
            'Color', [1 0.8 0.4]);

        % 검출 오버레이 (최대 200개만 표시, 밝기순)
        if app.params.show_overlay && app.cache.detection.n_detected > 0
            hold(ax2, 'on');
            dc = app.cache.detection.centroids;
            di = app.cache.detection.intensities;
            max_plot = min(200, size(dc, 1));
            [~, si] = sort(di, 'descend');
            dc = dc(si(1:max_plot), :);
            plot(ax2, dc(:,1), dc(:,2), 'ro', 'MarkerSize', 10, 'LineWidth', 1.5);
            hold(ax2, 'off');
        end
    end
end

%% ========== 메트릭 업데이트 ==========
function update_metrics(app)
    p = app.params;

    % FOV
    myu = p.pixel_size * 1e-6;
    f = p.focal_length * 1e-3;
    fovx = rad2deg(2 * atan((myu * p.res_w / 2) / f));
    fovy = rad2deg(2 * atan((myu * p.res_h / 2) / f));

    app.h.lbl_m_fov.Text = sprintf('FOV: %.2f x %.2f deg', fovx, fovy);
    app.h.lbl_m_res.Text = sprintf('Resolution: %d x %d px', p.res_w, p.res_h);
    app.h.lbl_m_ifov.Text = sprintf('IFOV: %.4f deg/px', rad2deg(atan(myu/f)));
    app.h.lbl_m_pointing.Text = sprintf('Pointing: RA=%.1f, DEC=%.1f, Roll=%.1f', ...
        p.ra, p.dec, p.roll);

    % 별 정보
    if ~isempty(app.cache.star_info)
        ns = app.cache.star_info.num_stars;
        nd = app.cache.detection.n_detected;
        app.h.lbl_m_stars.Text = sprintf('Stars in FOV: %d', ns);
        if ns > 0
            rate = nd/ns*100;
        else
            rate = 0;
        end
        app.h.lbl_m_detected.Text = sprintf('Detected: %d (%.1f%%)', nd, rate);
        app.h.lbl_m_snr.Text = sprintf('Peak SNR: %.1f dB', app.cache.snr_db);
        if ~isnan(app.cache.centroid_rms)
            app.h.lbl_m_rms.Text = sprintf('Centroid RMS: %.3f px (matched %d)', ...
                app.cache.centroid_rms, app.cache.centroid_matched);
        else
            app.h.lbl_m_rms.Text = 'Centroid RMS: N/A';
        end

        % 밝은 별 목록
        if ns > 0
            [sorted_mag, idx] = sort(app.cache.star_info.magnitudes);
            lines = 'Brightest stars: ';
            for i = 1:min(8, ns)
                j = idx(i);
                tc = app.cache.star_info.true_centroids(j,:);
                lines = [lines, sprintf('  #%d: mag=%.1f (%.0f,%.0f)', ...
                    i, sorted_mag(i), tc(1), tc(2))]; %#ok<AGROW>
            end
            app.h.lbl_m_brightest.Text = lines;
        end
    end

    % 처리 정보
    app.h.lbl_m_method.Text = sprintf('Method: %s', p.conv_method);
    if isfield(app.cache, 'sim_time')
        app.h.lbl_m_simtime.Text = sprintf('Sim time: %.2f s', app.cache.sim_time);
    end
    if isfield(app.cache, 'conv_time')
        app.h.lbl_m_convtime.Text = sprintf('Conv time: %.3f s', app.cache.conv_time);
    end
    app.h.lbl_m_exposure.Text = sprintf('Exp: %.1fms | Gain: %.0fx | QE: %.0f%%', ...
        p.exposure_ms, p.analog_gain, p.qe*100);
end

%% ========== FOV 실시간 표시 ==========
function update_fov_display(app)
    p = app.params;
    myu = p.pixel_size * 1e-6;
    f = p.focal_length * 1e-3;
    fovx = rad2deg(2 * atan((myu * p.res_w / 2) / f));
    fovy = rad2deg(2 * atan((myu * p.res_h / 2) / f));
    fovd = sqrt(fovx^2 + fovy^2);
    ifov = rad2deg(atan(myu / f));

    app.h.lbl_fov_h.Text = sprintf('H: %.2f deg', fovx);
    app.h.lbl_fov_v.Text = sprintf('V: %.2f deg', fovy);
    app.h.lbl_fov_d.Text = sprintf('Diagonal: %.2f deg', fovd);
    app.h.lbl_ifov.Text = sprintf('IFOV: %.4f deg/px', ifov);
end

%% ========== 노이즈 버짓 실시간 ==========
function update_noise_budget(app)
    p = app.params;
    exp_s = p.exposure_ms * 1e-3;
    total_gain = p.analog_gain * p.digital_gain;

    % 신호 추정 (Pogson)
    ref_flux = 96;  % photons/s for mag 6 (calibrated)
    ref_mag = 6.0;
    for mag = [1 3 6]
        pf = ref_flux * 10^(-0.4 * (mag - ref_mag));
        electrons = pf * exp_s * p.qe;
        adu = min(255, electrons * total_gain);
        switch mag
            case 1, app.h.lbl_sig1.Text = sprintf('1st mag: %.0f ADU (%.0f e-)', adu, electrons);
            case 3, app.h.lbl_sig3.Text = sprintf('3rd mag: %.0f ADU (%.0f e-)', adu, electrons);
            case 6, app.h.lbl_sig6.Text = sprintf('6th mag: %.0f ADU (%.0f e-)', adu, electrons);
        end
    end

    % 노이즈 성분
    sig6_e = ref_flux * exp_s * p.qe;  % 6등급 전자수
    shot_adu = sqrt(sig6_e) * total_gain;
    read_adu = p.read_noise * total_gain;
    dark_e = p.dark_current * exp_s;
    dark_adu = sqrt(dark_e) * total_gain;
    total_adu = sqrt(shot_adu^2 + read_adu^2 + dark_adu^2);

    app.h.lbl_nshot.Text = sprintf('Shot noise (6mag): %.1f ADU', shot_adu);
    app.h.lbl_nread.Text = sprintf('Read noise (ADU):  %.1f ADU', read_adu);
    app.h.lbl_ndark.Text = sprintf('Dark current (ADU): %.2f ADU', dark_adu);
    app.h.lbl_ntotal.Text = sprintf('Total noise RMS:   %.1f ADU', total_adu);

    % Exposure 라벨
    app.h.lbl_exp.Text = sprintf('%.1f ms', p.exposure_ms);
    app.h.lbl_gain.Text = sprintf('%.1fx', p.analog_gain);
end

%% ================================================================
%% =================== CALLBACKS ==================================
%% ================================================================

% --- Sensor callbacks ---
function cb_sensor(src, ~, param, fig)
    app = fig.UserData;
    val = src.Value;
    switch param
        case 'pixel_size'
            app.params.pixel_size = val;
            app.h.edit_pix.Value = val;
        case 'focal_length'
            app.params.focal_length = val;
            app.h.edit_fl.Value = val;
        case 'qe'
            app.params.qe = val;
            app.h.edit_qe.Value = val;
        case 'mag_limit'
            app.params.mag_limit = val;
            app.h.edit_mag.Value = val;
    end
    app.dirty.stage1 = true;
    fig.UserData = app;
    update_fov_display(app);
    update_noise_budget(app);
end

function cb_sensor_edit(src, ~, param, fig)
    app = fig.UserData;
    val = src.Value;
    switch param
        case 'pixel_size'
            app.params.pixel_size = val;
            app.h.slider_pix.Value = val;
        case 'focal_length'
            app.params.focal_length = val;
            app.h.slider_fl.Value = val;
        case 'qe'
            app.params.qe = val;
            app.h.slider_qe.Value = val;
        case 'mag_limit'
            app.params.mag_limit = val;
            app.h.slider_mag.Value = val;
    end
    app.dirty.stage1 = true;
    fig.UserData = app;
    update_fov_display(app);
    update_noise_budget(app);
end

function cb_res_preset(~, ~, w, h, fig)
    app = fig.UserData;
    app.params.res_w = w;
    app.params.res_h = h;
    app.h.edit_resw.Value = w;
    app.h.edit_resh.Value = h;
    app.dirty.stage1 = true;
    fig.UserData = app;
    update_fov_display(app);
end

function cb_res_changed(~, ~, fig)
    app = fig.UserData;
    app.params.res_w = round(app.h.edit_resw.Value);
    app.params.res_h = round(app.h.edit_resh.Value);
    app.dirty.stage1 = true;
    fig.UserData = app;
    update_fov_display(app);
end

% --- Exposure callbacks ---
function cb_exp_slider(src, ~, fig)
    app = fig.UserData;
    val_ms = 2^src.Value;
    val_ms = max(1, min(500, val_ms));
    app.params.exposure_ms = val_ms;
    app.dirty.stage1 = true;
    fig.UserData = app;
    update_noise_budget(app);
end

function cb_exp_preset(~, ~, val_ms, fig)
    app = fig.UserData;
    app.params.exposure_ms = val_ms;
    app.h.slider_exp.Value = log2(val_ms);
    app.dirty.stage1 = true;
    fig.UserData = app;
    update_noise_budget(app);
end

function cb_gain_preset(~, ~, gain, fig)
    app = fig.UserData;
    app.params.analog_gain = gain;
    app.h.slider_gain.Value = gain;
    app.dirty.stage1 = true;
    fig.UserData = app;
    update_noise_budget(app);
end

function cb_gain_adjust(~, ~, delta, fig)
    app = fig.UserData;
    new_gain = max(1, min(64, app.params.analog_gain + delta));
    app.params.analog_gain = new_gain;
    app.h.slider_gain.Value = new_gain;
    app.dirty.stage1 = true;
    fig.UserData = app;
    update_noise_budget(app);
end

function cb_gain_slider(src, ~, fig)
    app = fig.UserData;
    app.params.analog_gain = round(src.Value);
    app.dirty.stage1 = true;
    fig.UserData = app;
    update_noise_budget(app);
end

function cb_dgain(src, ~, fig)
    app = fig.UserData;
    app.params.digital_gain = src.Value;
    app.h.edit_dgain.Value = src.Value;
    app.dirty.stage1 = true;
    fig.UserData = app;
    update_noise_budget(app);
end

function cb_dgain_edit(src, ~, fig)
    app = fig.UserData;
    app.params.digital_gain = src.Value;
    app.h.slider_dgain.Value = src.Value;
    app.dirty.stage1 = true;
    fig.UserData = app;
    update_noise_budget(app);
end

% --- Noise callbacks ---
function cb_noise_toggle(src, ~, fig)
    app = fig.UserData;
    app.params.noise_enabled = src.Value;
    app.dirty.stage1 = true;
    fig.UserData = app;
end

function cb_noise_param(src, ~, param, fig)
    app = fig.UserData;
    switch param
        case 'dark_current'
            app.params.dark_current = src.Value;
            app.h.edit_dark.Value = src.Value;
        case 'read_noise'
            app.params.read_noise = src.Value;
            app.h.edit_read.Value = src.Value;
    end
    app.dirty.stage1 = true;
    fig.UserData = app;
    update_noise_budget(app);
end

function cb_noise_param_edit(src, ~, param, fig)
    app = fig.UserData;
    switch param
        case 'dark_current'
            app.params.dark_current = src.Value;
            app.h.slider_dark.Value = src.Value;
        case 'read_noise'
            app.params.read_noise = src.Value;
            app.h.slider_read.Value = src.Value;
    end
    app.dirty.stage1 = true;
    fig.UserData = app;
    update_noise_budget(app);
end

function cb_dark_preset(~, ~, val, fig)
    app = fig.UserData;
    app.params.dark_current = val;
    app.h.slider_dark.Value = val;
    app.h.edit_dark.Value = val;
    app.dirty.stage1 = true;
    fig.UserData = app;
    update_noise_budget(app);
end

function cb_scene_preset(~, ~, preset, fig)
    app = fig.UserData;
    switch preset
        case 'space'
            app.params.exposure_ms = 22;
            app.params.analog_gain = 16;
            app.params.digital_gain = 1.0;
            app.params.dark_current = 0.01;
            app.params.read_noise = 3;
            app.params.noise_enabled = true;
        case 'ground'
            app.params.exposure_ms = 100;
            app.params.analog_gain = 4;
            app.params.digital_gain = 1.0;
            app.params.dark_current = 0.1;
            app.params.read_noise = 3;
            app.params.noise_enabled = true;
        case 'low_noise'
            app.params.exposure_ms = 50;
            app.params.analog_gain = 8;
            app.params.digital_gain = 1.0;
            app.params.dark_current = 0;
            app.params.read_noise = 0;
            app.params.noise_enabled = false;
        case 'high_noise'
            app.params.exposure_ms = 22;
            app.params.analog_gain = 64;
            app.params.digital_gain = 2.0;
            app.params.dark_current = 1.0;
            app.params.read_noise = 15;
            app.params.noise_enabled = true;
    end
    % UI 동기화
    app.h.slider_exp.Value = log2(max(1, app.params.exposure_ms));
    app.h.slider_gain.Value = app.params.analog_gain;
    app.h.slider_dgain.Value = app.params.digital_gain;
    app.h.edit_dgain.Value = app.params.digital_gain;
    app.h.slider_dark.Value = app.params.dark_current;
    app.h.edit_dark.Value = app.params.dark_current;
    app.h.slider_read.Value = app.params.read_noise;
    app.h.edit_read.Value = app.params.read_noise;
    app.h.chk_noise.Value = app.params.noise_enabled;

    app.dirty.stage1 = true;
    fig.UserData = app;
    update_noise_budget(app);
end

% --- Observation callbacks ---
function cb_obs(src, ~, param, fig)
    app = fig.UserData;
    switch param
        case 'ra'
            app.params.ra = src.Value;
            app.h.edit_ra.Value = src.Value;
        case 'dec'
            app.params.dec = src.Value;
            app.h.edit_dec.Value = src.Value;
        case 'roll'
            app.params.roll = src.Value;
            app.h.edit_roll.Value = src.Value;
    end
    app.dirty.stage1 = true;
    fig.UserData = app;
end

function cb_obs_edit(src, ~, param, fig)
    app = fig.UserData;
    switch param
        case 'ra'
            app.params.ra = src.Value;
            app.h.slider_ra.Value = src.Value;
        case 'dec'
            app.params.dec = src.Value;
            app.h.slider_dec.Value = src.Value;
        case 'roll'
            app.params.roll = src.Value;
            app.h.slider_roll.Value = src.Value;
    end
    app.dirty.stage1 = true;
    fig.UserData = app;
end

function cb_obs_preset(~, ~, ra, dec, fig)
    app = fig.UserData;
    if ra < 0  % Random
        ra = rand() * 360;
        dec = rand() * 180 - 90;
    end
    app.params.ra = ra;
    app.params.dec = dec;
    app.h.slider_ra.Value = ra;
    app.h.edit_ra.Value = ra;
    app.h.slider_dec.Value = dec;
    app.h.edit_dec.Value = dec;
    app.dirty.stage1 = true;
    fig.UserData = app;
end

% --- Processing callbacks ---
function cb_method_changed(src, ~, fig)
    app = fig.UserData;
    app.params.conv_method = src.Value;
    app.h.panel_weights.Visible = strcmp(src.Value, 'optimal');
    app.dirty.stage2 = true;
    fig.UserData = app;
end

function cb_weight_changed(~, ~, fig)
    app = fig.UserData;
    w_R = app.h.slider_wR.Value;
    w_G = app.h.slider_wG.Value;
    w_B = app.h.slider_wB.Value;
    total = w_R + w_G + w_B;

    app.h.lbl_wR.Text = sprintf('%.4f', w_R);
    app.h.lbl_wG.Text = sprintf('%.4f', w_G);
    app.h.lbl_wB.Text = sprintf('%.4f', w_B);
    app.h.lbl_wsum.Text = sprintf('Sum: %.3f', total);

    app.params.opt_weights = [w_R, w_G, w_B];
    app.dirty.stage2 = true;
    fig.UserData = app;
end

function cb_normalize_weights(~, ~, fig)
    app = fig.UserData;
    w = app.params.opt_weights;
    total = sum(w);
    if total > 0
        w = w / total;
    end
    app.params.opt_weights = w;
    app.h.slider_wR.Value = w(1);
    app.h.slider_wG.Value = w(2);
    app.h.slider_wB.Value = w(3);
    app.h.lbl_wR.Text = sprintf('%.4f', w(1));
    app.h.lbl_wG.Text = sprintf('%.4f', w(2));
    app.h.lbl_wB.Text = sprintf('%.4f', w(3));
    app.h.lbl_wsum.Text = sprintf('Sum: %.3f', sum(w));
    app.dirty.stage2 = true;
    fig.UserData = app;
end

function cb_reset_weights(~, ~, fig)
    app = fig.UserData;
    w = [0.4544, 0.3345, 0.2111];
    app.params.opt_weights = w;
    app.h.slider_wR.Value = w(1);
    app.h.slider_wG.Value = w(2);
    app.h.slider_wB.Value = w(3);
    app.h.lbl_wR.Text = sprintf('%.4f', w(1));
    app.h.lbl_wG.Text = sprintf('%.4f', w(2));
    app.h.lbl_wB.Text = sprintf('%.4f', w(3));
    app.h.lbl_wsum.Text = sprintf('Sum: %.3f', sum(w));
    app.dirty.stage2 = true;
    fig.UserData = app;
end

function cb_detect_param(src, ~, param, fig)
    app = fig.UserData;
    val = round(src.Value);
    switch param
        case 'threshold'
            app.params.threshold = val;
            app.h.edit_thresh.Value = val;
        case 'min_area'
            app.params.min_area = val;
            app.h.edit_minarea.Value = val;
    end
    app.dirty.stage3 = true;
    fig.UserData = app;
end

function cb_detect_param_edit(src, ~, param, fig)
    app = fig.UserData;
    val = round(src.Value);
    switch param
        case 'threshold'
            app.params.threshold = val;
            app.h.slider_thresh.Value = val;
        case 'min_area'
            app.params.min_area = val;
            app.h.slider_minarea.Value = val;
    end
    app.dirty.stage3 = true;
    fig.UserData = app;
end

function cb_display_opt(~, ~, fig)
    app = fig.UserData;
    app.params.show_overlay = app.h.chk_overlay.Value;
    app.params.show_labels = app.h.chk_labels.Value;
    fig.UserData = app;
    % 이미지만 다시 그리기 (시뮬레이션 불필요)
    if ~isempty(app.cache.gray_ideal)
        update_images(app);
    end
end

% --- Action buttons ---
function cb_simulate(~, ~, fig)
    run_simulation(fig, true);
end

function cb_quick_update(~, ~, fig)
    app = fig.UserData;
    if isempty(app.cache.bayer_img)
        app.h.lbl_status.Text = 'No cached data. Run [Simulate] first!';
        fig.UserData = app;
        return;
    end
    app.dirty.stage1 = false;
    app.dirty.stage2 = true;
    fig.UserData = app;
    run_simulation(fig, false);
end

function cb_export(~, ~, fig)
    app = fig.UserData;
    if isempty(app.cache.gray_ideal)
        app.h.lbl_status.Text = 'No images to export. Run [Simulate] first!';
        return;
    end
    script_dir = fileparts(mfilename('fullpath'));
    out_dir = fullfile(script_dir, 'output');
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end

    imwrite(uint8(min(255, app.cache.gray_ideal)), ...
        fullfile(out_dir, 'gui_ideal.png'));
    imwrite(app.cache.gray_converted, ...
        fullfile(out_dir, sprintf('gui_converted_%s.png', app.params.conv_method)));

    app.h.lbl_status.Text = sprintf('Exported to %s', out_dir);
    fig.UserData = app;
end
