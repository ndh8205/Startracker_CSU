function result = detect_stars_simple(img, threshold, min_area)
% DETECT_STARS_SIMPLE 간단한 별 검출 (threshold + connected component)

if nargin < 2, threshold = 20; end
if nargin < 3, min_area = 3; end

img = double(img);

% 배경 추정 (중앙값)
bg = median(img(:));

% 임계값 적용
binary = img > (bg + threshold);

% Connected component 분석
cc = bwconncomp(binary);
stats = regionprops(cc, img, 'Centroid', 'Area', 'MaxIntensity', 'MeanIntensity');

% 최소 크기 필터링
valid_idx = find([stats.Area] >= min_area);

result.n_detected = length(valid_idx);
result.centroids = zeros(length(valid_idx), 2);
result.intensities = zeros(length(valid_idx), 1);

for i = 1:length(valid_idx)
    result.centroids(i, :) = stats(valid_idx(i)).Centroid;
    result.intensities(i) = stats(valid_idx(i)).MaxIntensity;
end
end
