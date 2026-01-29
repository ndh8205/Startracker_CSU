function snr = calculate_snr(measured, reference)
% CALCULATE_SNR SNR 계산 (dB)
%
% SNR = 10 * log10(signal_power / noise_power)

% 크기 맞춤
[h1, w1] = size(measured);
[h2, w2] = size(reference);

if h1 ~= h2 || w1 ~= w2
    reference = imresize(reference, [h1, w1]);
end

measured = double(measured);
reference = double(reference);

signal_power = mean(reference(:).^2);
noise_power = mean((measured(:) - reference(:)).^2);

if noise_power < 1e-10
    snr = 100;
else
    snr = 10 * log10(signal_power / noise_power);
end
end
