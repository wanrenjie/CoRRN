clear all
%% settings
foldertrain = '/home/rjwan/Datageneration/Train2';
folderreflection1 = '/home/rjwan/Datageneration/rimage';
dest = '/home/rjwan/Datageneration/imagedata';

%savepath = 'trainA5.h5';
%% generate data
filepathstrain = dir(fullfile(foldertrain,'*.jpg'));
filepathsreflection = dir(fullfile(folderreflection1,'*.jpg'));


weightR = [0.7,0.8,0.9];
weightB = [0.8,0.9,0.7]; %You can try more weighting setting to get different results
count = 1;
for i = 1:9
    j = 1
    while j <= length(filepathstrain)
		
		image = (imread(fullfile(foldertrain,filepathstrain(j).name)));

        image = imresize(image, [96,128]);
        im_label = double(image);
        
        order = randi([1, length(filepathsreflection)]);
        
        layer2 = double((imread(fullfile(folderreflection1,filepathsreflection(order).name))));
        layer2 = imresize(layer2, [96,128]);
        [hei,wid,channel] = size(im_label);

        layer2 = imgaussfilt(layer2, 0.4);
        num = randi(3);
        
        im_input = weightB(num)*im_label + weightR(num)*layer2;

        edgeB = gradient(double(rgb2gray(image)));
        %edgeinput = gradient(rgb2gray(im_input));
        name = sprintf('%05d', count);
        foldername = num2str(name);
        namem1 = [foldername,'_m.jpg'];
        namer1 = [foldername,'_r.jpg'];
        nameg1 = [foldername,'_g.mat'];
        nameb1 = [foldername,'_b.jpg']
        mkdir(fullfile(dest, name));
        
        imwrite(uint8(im_input), fullfile(dest, name, namem1));
        imwrite(uint8(1*im_label), fullfile(dest, name, nameb1));
        imwrite(uint8(weightR(num)*layer2), fullfile(dest, name, namer1));
        save(fullfile(dest, name, nameg1), 'edgeB');
        
        j = j + 1;
        count = count + 1;
    end
end    








