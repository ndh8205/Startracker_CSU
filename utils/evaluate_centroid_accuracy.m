function result = evaluate_centroid_accuracy(detection, true_centroids, match_radius)
% EVALUATE_CENTROID_ACCURACY Centroid 정확도 평가

if nargin < 3, match_radius = 5.0; end

result.n_matched = 0;
result.errors = [];

if detection.n_detected == 0 || isempty(true_centroids)
    result.rms_error = NaN;
    return;
end

for i = 1:size(true_centroids, 1)
    true_pos = true_centroids(i, :);

    % 가장 가까운 검출 별 찾기
    distances = sqrt(sum((detection.centroids - true_pos).^2, 2));
    [min_dist, ~] = min(distances);

    if min_dist < match_radius
        result.n_matched = result.n_matched + 1;
        result.errors = [result.errors; min_dist];
    end
end

if isempty(result.errors)
    result.rms_error = NaN;
else
    result.rms_error = sqrt(mean(result.errors.^2));
end
end
