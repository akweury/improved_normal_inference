path = 'SyntheticDataSet\CapturedData\00000.';
%path = 'RealDataSet\00000.';

%load image
C.image = imread([path, 'image0.png']);
C.image = cat(3, C.image, C.image, C.image);

%load data, containing calibration, min/max depths
fid = fopen([path, 'data0.json'], 'r'); 
str = char(fread(fid,inf)'); 
fclose(fid); 
C.data = jsondecode(str);

%load depth map
C.depth = double(imread([path, 'depth0.png']));
C.mask = C.depth ~= 0;
C.depth = C.mask .* (C.depth*(C.data.maxDepth-C.data.minDepth)/65535 + C.data.minDepth);

%compute vertex map from depth map and calibration
C.vertex = depth2vertex(C.depth, C.data.K);

%load normals in tangent space and transform to normal space, 
%apply rotation, if camera is rotated
normal_tan = double(imread([path, 'normal0.png']));
normalOrig = cat(3, C.mask, C.mask, -C.mask) .* (cat(3, normal_tan(:,:,1), normal_tan(:,:,2), normal_tan(:,:,3))/255*2 - 1);
C.normal = normalOrig;
C.normal(:,:,1) = C.data.R(1,1)*normalOrig(:,:,1) + C.data.R(2,1)*normalOrig(:,:,2) + C.data.R(3,1)*normalOrig(:,:,3);
C.normal(:,:,2) = C.data.R(1,2)*normalOrig(:,:,1) + C.data.R(2,2)*normalOrig(:,:,2) + C.data.R(3,2)*normalOrig(:,:,3);
C.normal(:,:,3) = C.data.R(1,3)*normalOrig(:,:,1) + C.data.R(2,3)*normalOrig(:,:,2) + C.data.R(3,3)*normalOrig(:,:,3);

%compute point clouds and visualizations of the camera and the light source
C.pointCloud = RGBDN2pointCloud(C.image, C.depth, C.normal, C.data);

camera = camera_visualization/4;
light = light_visualization/8;
camera = C.data.R'*camera - repmat(C.data.R'*C.data.t,1,size(camera,2));
light = light + repmat(C.data.lightPos,1,size(light,2));
camera = [camera; zeros(size(camera)); 255*ones(size(camera))]; 
camera(8:9,:) = 0;
light = [light; zeros(size(light)); 255*ones(size(light))];
light(9,:) = 0;

savePLY([path, 'pointcloud0.ply'], [C.pointCloud, camera, light]);

function vertex = depth2vertex(depth, K)

    [X,Y] = meshgrid(1:size(depth,2), 1:size(depth,1));
    X = X - K(1,3);
    Y = Y - K(2,3);
    Z = ones(size(depth)) * K(1,1);
    Dir = cat(3,X,Y,Z);
    vertex = Dir .* repmat(depth ./ sqrt(sum(Dir.^2,3)),1,1,3);
end

function pointCloud = RGBDN2pointCloud(im, depth, normal, P)

    camOrig = -P.R' * P.t;
    %triangulate point cloud from cameras and matches
    N = sum(sum(depth ~= 0));
    pointCloud = zeros(9, N);
    n = 1;
    for x = 1 : size(depth, 2)
        for y = 1 : size(depth, 1)
            if depth(y, x) ~= 0
                dir = [x - P.K(1,3); y - P.K(2,3); P.K(1,1)];
                pointCloud(1:3,n) = camOrig + P.R' * dir * depth(y, x) / norm(dir);
                pointCloud(4:6,n) = squeeze(normal(y,x,:));
                pointCloud(7:9,n) = squeeze(im(y,x,:));
                n = n+1;
            end
        end
    end
end

function savePLY( dest, points )

    N = size(points, 2);
    fileID = fopen(dest, 'w');
    fprintf(fileID, 'ply\n');
    fprintf(fileID, 'format binary_little_endian 1.0\n');
    fprintf(fileID, ['element vertex ', num2str(N), '\n']);
    fprintf(fileID, 'property float x\n');
    fprintf(fileID, 'property float y\n');
    fprintf(fileID, 'property float z\n');
    fprintf(fileID, 'property float nx\n');
    fprintf(fileID, 'property float ny\n');
    fprintf(fileID, 'property float nz\n');
    fprintf(fileID, 'property uchar red\n');
    fprintf(fileID, 'property uchar green\n');
    fprintf(fileID, 'property uchar blue\n');
    fprintf(fileID, 'end_header\n');

    colors = 255 * ones(3, N);
    normals = zeros(3, N);
    if size(points, 1) == 6
        colors = points(4:6,:);
    end
    if size(points, 1) == 9
        normals = points(4:6,:);
        colors = points(7:9,:);
    end

    points_8bit = reshape(typecast(single(reshape(points(1:3,:), 3*N, 1)), 'uint8'), 12, N);
    normals_8bit = reshape(typecast(single(reshape(normals, 3*N, 1)), 'uint8'), 12, N);
    combined = reshape([points_8bit; normals_8bit; colors], 27*N, 1);

    fwrite(fileID, combined, 'uint8');
    fclose(fileID);
end

function C = loadData(path, i, net)
     
    %compute point clouds
    camera = camera_visualization/4;
    light = light_visualization/8;
    camera = C.data.R'*camera - repmat(C.data.R'*C.data.t,1,size(camera,2));
    light = light + repmat(C.data.lightPos,1,size(light,2));
    camera = [camera; zeros(size(camera)); 255*ones(size(camera))]; 
    camera(8:9,:) = 0;
    light = [light; zeros(size(light)); 255*ones(size(light))];
    light(9,:) = 0;
    
    C.pointCloud = RGBDN2pointCloud(C.image, C.depth, C.normal, C.data);
    savePLY([path, 'pointcloud', num2str(i), net, '.ply'], [C.pointCloud, camera, light]);
end

function points = camera_visualization(  )

points = [];

for px = -0.5 : 0.01 : 0.5
    for py = -0.5 : 0.01 : 0.5
        for pz = -0.5 : 0.01 : 0.5
            p = [px; py; pz];
            p_sorted = sort(abs(p));
            if p_sorted(2) == 0.5 && p_sorted(3) == 0.5
                points = [points, p];
            end
        end
    end
end
for px = -0.3 : 0.01 : 0.3
    for py = -0.3 : 0.01 : 0.3
        for pz = -0.3 : 0.01 : 0.3
            p = [px; py; pz];
            p_sorted = sort(abs(p));
            if p_sorted(2) == 0.3 && p_sorted(3) == 0.3
                p = [px; py; pz+0.8];
                points = [points, p];
            end
        end
    end
end

end

function points = projector_visualization(  )

points = [];

for px = -0.5 : 0.01 : 0.8
    for py = -0.5 : 0.01 : 0.5
        for pz = -0.5 : 0.01 : 0.5
            p = [px; py; pz];
            p_sorted = sort(abs(p));
            if p_sorted(2) == 0.5 && p_sorted(3) == 0.5
                points = [points, p];
            end
        end
    end
end
points(2,:) = points(2,:)/2;
points(1,:) = points(1,:)+0.25;
points(3,:) = points(3,:)*0.8 + 0.2;

for px = -0.2 : 0.01 : 0.2
    for py = -0.2 : 0.01 : 0.2
        for pz = -0.2 : 0.01 : 0.2
            p = [px; py; pz];
            p_sorted = sort(abs(p));
            if p_sorted(2) == 0.2 && p_sorted(3) == 0.2
                p = [px; py; pz+0.8];
                points = [points, p];
            end
        end
    end
end

end

function points = light_visualization(  )

points = [];

for px = -0.5 : 0.1 : 0.5
    for py = -0.5 : 0.1 : 0.5
        for pz = -0.5 : 0.1 : 0.5
            p = [px; py; pz];
            if norm(p) > 0
                points = [points, p/norm(p)];
            end
        end
    end
end

end