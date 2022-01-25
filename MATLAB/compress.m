function [outRep] = compress(im, lambda, noQuantize)

% [outRep] = compress(im, lambda, noQuantize)
%
% Lossy compression of an image [Balle,Laparra,Simoncelli, ICLR2017].
% IM is an image (matrix) of grayscale pixel values. LAMBDA determines
% the amount of compression rate (tradeoff between bits and quality),
% and can only take values 2^k for k in the range [5:12]. OUTREP is a
% tensor (3D matrix) containing 128 channels that form the coded
% representation of the image.  By default, these are quantized to
% integer values, but if NOQUANTIZE (optional) is provided and
% non-zero, the final quantization step is skipped and floating point
% channel values are returned.
%
% Relies on convolution code from matlabPyrTools, available at:
%    http://www.cns.nyu.edu/~lcv/software.php 
%
% Published in:
%   End-to-end optimized image compression
%   J BallÃ©, V Laparra and E P Simoncelli
%   Int'l. Conference on Learning Representations (ICLR2017), April 2017. 
%   https://arxiv.org/abs/1611.01704
%
% See http://www.cns.nyu.edu/~lcv/iclr2017/ for further information.
% 
% Released July, 2017.

% Verify that image is grayscale, double float, in range [0,1]
if ((length(size(im))) ~= 2)
    error('Image argument must be 2-dimensional (grayscale)');
end
im = double(im); 
if (min(im(:))<0)
    warning('IMAGE contains negative values - adjusting by adding constant offset %.2f.',-min(im(:)));
    im = im-min(im(:));
end
if (max(im(:))>1)
    warning('Rescaling IMAGE by factor %.2f to fit within range [0,1].',max(im(:)));
    im = im/max(im(:));
end

% Coder parameters have only been optimized for lambda=2^n, n an integer in [5,12].
rl =  2.^min(max(round(log2(lambda)),5),12);  
if (abs(rl-lambda)>0.1)
    warning('Rounding lambda value %.2f to %d', lambda, rl);
end
lambda = rl;

if (exist('noQuantize','var') ~= 1)
    noQuantize = 0;  
end

% Pad image size to 41 + a multiple of 16: yields output size size(im)/16 for 'valid' convolutions
pad = 41 + 16 + 16*ceil(size(im)/16) - size(im);  %***NOTE: hardwired, assumes [4,2,2] subsampling
bdry = [floor(pad(1)/2), floor(pad(2)/2), ceil(pad(1)/2), ceil(pad(2)/2)];
im = [im(bdry(1):-1:1,bdry(2):-1:1), im(bdry(1):-1:1,:), im(bdry(1):-1:1,end:-1:(end-bdry(4)+1)); im(:,bdry(2):-1:1), im, im(:,end:-1:(end-bdry(4)+1)); im(end:-1:(end-bdry(3)+1),bdry(2):-1:1), im(end:-1:(end-bdry(3)+1),:), im(end:-1:(end-bdry(3)+1),end:-1:(end-bdry(4)+1))];

% File directory containing parameter files, relative to m-file source
% code of the current function
myPath = fileparts(mfilename('fullpath'));
paramDir = sprintf('%s/parameters/gray/lambda-%06d/', myPath, lambda);

inRep = im;                             % Initialize input representation to input image
inDims = size(im);
inDepth = 1;                            % "depth" = number of channels 

for stage = 0:2
    load(sprintf('%sanalysis-%02d.mat', paramDir, stage), 'h', 'strides', 'c', 'beta', 'gamma');
    filters = double(h); clear h; fDims = [size(filters,3),size(filters,4)];
    downSampling = double(strides); clear strides;
    const = double(c); clear c;
    beta = double(beta); gamma = double(gamma);

    if (inDepth ~= size(filters,2))
        error(sprintf('Stage %d: depth of input %d does not match filter set',stage,inDepth));
    end
    outDims = floor((inDims-(fDims-1)+(downSampling-1))./downSampling);
    inStart = 1+(fDims-1)/2; % first valid convolution location
    inStop = inDims-(fDims-1)/2; % last valid convolution location
    outDepth = size(filters,1); 

    outRep = zeros([outDims, outDepth]);
    for n=1:outDepth
        for m=1:inDepth
            filt = squeeze(filters(n,m,:,:));  edges = 'reflect1';
            % Correlate and downsample, with params chosen to act like matlab's 'valid' convolution
            outRep(:,:,n) = outRep(:,:,n) + corrDn(inRep(:,:,m), filt, edges, downSampling, inStart, inStop);
        end
        outRep(:,:,n) = outRep(:,:,n) + const(n);
    end

    % divisive normalization
    norm2 = reshape(outRep.^2, [prod(outDims), outDepth]);
    norm2 = norm2 * gamma + ones(size(norm2,1),1) * beta;
    outRep = outRep ./ reshape(sqrt(norm2), size(outRep));

    inRep = outRep;                     % set input for next stage to output of this stage
    inDims = outDims;
    inDepth = outDepth;
end

stage = 3;                              % final stage: additive constant only
load(sprintf('%sanalysis-%02d.mat', paramDir, stage), 'c');
const = double(c); clear c;

for n=1:outDepth
    outRep(:,:,n) = outRep(:,:,n) + const(n);
end

if (~noQuantize)
    outRep = round(outRep);
end

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% TEST

clear all
im = double(pgmRead('einstein.pgm')/255);
%im = (im-min(im(:)))/(max(im(:))-min(im(:)));  % image must lie in range [0,1]

lambda = 2^9;                           % choose an integer exponent in range [5,12]
                                        
codedIm = compress(im, lambda);

% Visualize the coded channels
figure(2); clf
chSz = [size(codedIm,1),size(codedIm,2)];
nch = size(codedIm,3); nchy = 2.^floor(log2(nch)/2); nchx = nch/nchy;
chCols = reshape(codedIm,[prod(chSz), nch]);  % matrix with channels in columns
chCols = chCols ./ (ones(size(chCols,1),1)*(1+max(chCols)-min(chCols)));  % normalize
showIm(col2im(chCols, chSz, [nchy,nchx].*chSz, 'distinct'));

% Rough estimate of bitrate:
binCtrs = [floor(min(codedIm(:))):ceil(max(codedIm(:)))];
H = hist(reshape(codedIm,[prod(chSz), nch]), binCtrs);
bits=0;
for k=1:size(H,2)
    nhist = H(:,k);  nhist = nhist(find(nhist>0.5));  nhist = nhist/sum(nhist);
    bits = bits - (nhist'*log2(nhist))*prod(chSz);
end
entropy = bits/prod(size(im));

res =  uncompress(codedIm, lambda, size(im));

% Peak signal-to-noise ratio (PSNR):
psnr = 10*log10( 1/mean((im(:)-res(:)).^2) );

% View compressed image, next to original 
figure(1); clf
ttl = sprintf('\\lambda=%d,  R=%.3f b/px,  PSNR=%.1fdB', lambda,entropy,psnr);
showIm(im+sqrt(-1)*res,[0,1],2,ttl);       % display (im,result) at same
                                           % intensity scale, zoomed by 2

%% Compute/plot a rate-distortion curve, like the ones in the paper
figure(1); clf
lambdas = 2.^[5:12];
resIms = zeros([size(im), length(lambdas)]);  
entropies = zeros(1,length(lambdas));  psnrs = zeros(1,length(lambdas)); 
for n=[1:length(lambdas)]
    lambda = lambdas(n);
    codedIm = compress(im,lambda);  chSz = size(codedIm);
    res =  uncompress(codedIm, lambda, size(im));
    binCtrs = [floor(min(codedIm(:))):ceil(max(codedIm(:)))];
    H = hist(reshape(codedIm,[chSz(1)*chSz(2), chSz(3)]), binCtrs);
    bits=0;
    for k=1:size(H,2)
        nhist = H(:,k);  nhist = nhist(find(nhist>0.5));  nhist = nhist/sum(nhist);
        bits = bits - (nhist'*log2(nhist))*chSz(1)*chSz(2);
    end
    entropy = bits/prod(size(im));
    psnr =  10*log10(1/mean((im(:)-res(:)).^2));
    resIms(:,:,n) = res; entropies(n) = entropy; psnrs(n) = psnr;
    ttl = sprintf('lambda=%d,  R=%.3f b/px,  PSNR=%.1fdB', lambda,entropy,psnr);
    figure(1); showIm(im+sqrt(-1)*res,[0,1],1,ttl); pause(0.1);
end

fprintf(1,'lambda=%04d,  R=%.3f b/px,  PSNR=%.1fdb\n', [lambdas; entropies; psnrs]);

figure(2); clf
plot(entropies, psnrs, '.-r'); grid on
xlabel('Rate (bits/pixel)'); ylabel('PSNR');