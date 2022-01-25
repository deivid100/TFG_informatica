% NUMERO DE FILTROS
%{
bpp = 1:5;
bpp(1)= (3002525*8)/(1920*1065);
bpp(2)=(3378322*8)/(1920*1065);
bpp(3)=(3420359*8)/(1920*1065);
bpp(4)=(3495108*8)/(1920*1065);
bpp(5)=(3578133*8)/(1920*1065);
psnr = [32.036045, 32.570927, 33.353851, 33.675275, 34.170308];

bpp2 = 1:5;
bpp2(1)= (877719*8)/(1024*576);
bpp2(2)= (996227*8)/(1024*576);
bpp2(3)= (970475*8)/(1024*576);
bpp2(4)=(987078*8)/(1024*576);
bpp2(5)=(1011719*8)/(1024*576);
psnr2 = [32.815508, 33.337969, 34.516386, 34.985190, 35.580581];

bpp3 = 1:5;
bpp3(1)= (1815678*8)/(1350*900);
bpp3(2)= (2078710*8)/(1350*900);
bpp3(3)= (2115220*8)/(1350*900);
bpp3(4)= (2157296*8)/(1350*900);
bpp3(5)= (2200556*8)/(1350*900);
psnr3 = [31.827060, 32.801836, 34.030458, 34.545020, 35.354703];

plot(psnr, bpp, 'linewidth',3);
hold on
plot(psnr2,bpp2, 'linewidth',3)
hold on
plot(psnr3,bpp3, 'linewidth',3)
hold off
xlabel('PSNR','Fontsize',24);
ylabel('BITRATE', 'Fontsize',24);
title('NUMERO DE FILTROS')
legend({'Figura 2','Figura 3','Figura 4'},'Location','southeast',fontsize=24)
%}

%{
% TAMAÑO DEL FILTRO
bpp = 1:4;
bpp(1)= (3473355*8)/(1920*1065);
bpp(2)=(3420359*8)/(1920*1065);
bpp(3)=(3516735*8)/(1920*1065);
bpp(4)=(3503817*8)/(1920*1065);
psnr = [33.301138, 33.353851, 33.001163, 33.146949];

bpp2 = 1:4;
bpp2(1)= (992183*8)/(1024*576);
bpp2(2)= (970475*8)/(1024*576);
bpp2(3)= (1009295*8)/(1024*576);
bpp2(4)=(1012676*8)/(1024*576);
psnr2 = [34.350391, 34.516386, 34.028330, 34.192789];

bpp3 = 1:4;
bpp3(1)= (2144736*8)/(1350*900);
bpp3(2)= (2115220*8)/(1350*900);
bpp3(3)= (2180482*8)/(1350*900);
bpp3(4)= (2167907*8)/(1350*900);
psnr3 = [34.078758, 34.030458, 33.663822, 33.753207];

plot(psnr, bpp, 'linewidth',3);
hold on
plot(psnr2,bpp2, 'linewidth',3)
hold on
plot(psnr3,bpp3, 'linewidth',3)
hold off
xlabel('PSNR','Fontsize',24);
ylabel('BITRATE', 'Fontsize',24);
title('TAMAÑO DE FILTRO')
legend({'Figura 2','Figura 3','Figura 4'},'Location','southeast',fontsize=24)
%}

%NUMERO DE CAPAS
%{
bpp = 1:4;
bpp(1)= (3706773*8)/(1920*1065);
bpp(2)=(3420359*8)/(1920*1065);
bpp(3)=(2847751*8)/(1920*1065);
bpp(4)=(2382135*8)/(1920*1065);
psnr = [40.115888,33.353851,31.440910,30.333372];


bpp2 = 1:4;
bpp2(1)= (1037194*8)/(1024*576);
bpp2(2)= (970475*8)/(1024*576);
bpp2(3)= (852455*8)/(1024*576);
bpp2(4)=(741346*8)/(1024*576);
psnr2 = [42.064066,34.516386,32.122240,30.479883
];

bpp3 = 1:4;
bpp3(1)= (2266301*8)/(1350*900);
bpp3(2)= (2115220*8)/(1350*900);
bpp3(3)= (1713319*8)/(1350*900);
bpp3(4)= (1458326*8)/(1350*900);
psnr3 = [41.588315,34.030458,31.141286,30.041457
];

plot(psnr, bpp, 'linewidth',3);
hold on
plot(psnr2,bpp2, 'linewidth',3)
hold on
plot(psnr3,bpp3, 'linewidth',3)
hold off
xlabel('PSNR','Fontsize',24);
ylabel('BITRATE', 'Fontsize',24);
title('NÚMERO DE CAPAS')
legend({'Figura 2','Figura 3','Figura 4'},'Location','southeast',fontsize=24)
%}


%Stride
%{
bpp = 1:2;
bpp(1)= (3420359*8)/(1920*1065);
bpp(2)=(2141351*8)/(1920*1065);
psnr = [33.353851,29.533770
];


bpp2 = 1:2;
bpp2(1)= (970475*8)/(1024*576);
bpp2(2)= (634908*8)/(1024*576);

psnr2 = [34.516386,29.645231
];

bpp3 = 1:2;
bpp3(1)= (2115220*8)/(1350*900);
bpp3(2)= (1305963*8)/(1350*900);
psnr3 = [34.030458,29.241375
];

plot(psnr, bpp, 'linewidth',3);
hold on
plot(psnr2,bpp2, 'linewidth',3)
hold on
plot(psnr3,bpp3, 'linewidth',3)
hold off
xlabel('PSNR','Fontsize',24);
ylabel('BITRATE', 'Fontsize',24);
title('STRIDE')
legend({'Figura 2','Figura 3','Figura 4'},'Location','southeast',fontsize=24)
%}

%LAMBDA

bpp = 1:5;
bpp(1)= (2969821*8)/(1920*1065);
bpp(2)=(3403439*8)/(1920*1065);
bpp(3)=(3486266*8)/(1920*1065);
bpp(4)=(3498798*8)/(1920*1065);
bpp(5)=(3502679*8)/(1920*1065);

psnr = [31.428967,32.898896,33.288423,33.316628,33.340270
];

bpp2 = 1:5;
bpp2(1)= (887923*8)/(1024*576);
bpp2(2)= (963340*8)/(1024*576);
bpp2(3)= (968300*8)/(1024*576);
bpp2(4)=(970904*8)/(1024*576);
bpp2(5)=(986274*8)/(1024*576);

psnr2 = [31.698393,33.848279,34.425413,34.454924,34.495244
];

bpp3 = 1:5;
bpp3(1)= (1824675*8)/(1350*900);
bpp3(2)= (2098556*8)/(1350*900);
bpp3(3)= (2155270*8)/(1350*900);
bpp3(4)= (2177712*8)/(1350*900);
bpp3(5)= (2185439*8)/(1350*900);

psnr3 = [31.077134,33.381129,33.979686,33.966671,34.020454
];

plot(psnr, bpp, 'linewidth',3);
hold on
plot(psnr2,bpp2, 'linewidth',3)
hold on
plot(psnr3,bpp3, 'linewidth',3)
hold off
xlabel('PSNR','Fontsize',24);
ylabel('BITRATE', 'Fontsize',24);
title('MULTIPLICADOR LAGRANGE')
legend({'Figura 2','Figura 3','Figura 4'},'Location','southeast',fontsize=24)
