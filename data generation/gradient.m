function [ grad ] = gradient( img )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    [Gx, Gy] = imgradientxy(img);
    grad = sqrt(Gx.^2+Gy.^2);
end

