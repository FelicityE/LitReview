%% Setup
close all;
pkg load statistics
pkg load signal
pkg load financial

% Code/Ploting Controls
plotChanVClass = false; % for each dataset, plot all channels over time
plotAllPCvTime = false; % for each dataset, plot PC over time
plotPCoriVt = false; % for D1 to D7, plot corrisponding rotated dataset in 3D over time 
plot3DD5chan = false; % Plotting D5 in 3D PCA over time *Since channel has no corresponding rotated dataset
plotD2D7G2G3 = false; % Plot compare D2 G2 & G3 to D7 G2 & G3
windowShape = false; % Running window shapes 
  shapePlots = true; % plotting shapes
windowPeaks = true; % Running Window peaks
PCA = false;
LDA = false;


plotN = 0;
pod = 0.1; %Percent of dataset to plot/process


% Load data
if(exist("MLR_EMG_data.mat","file"))
  load MLR_EMG_data.mat
  dataN = dataset.nFiles;
else
  dataN = LoadData("EMG_data_for_gestures-master", "txt", "MLR_EMG_data.mat");
  load MLR_EMG_data.mat;
endif

% select number of datasets to process
Ndata = round(dataN*0.1);
NG = 6; %number of guestures
NChan = 8; %number of channels


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for each dataset, plot all channels over time
if(plotChanVClass)
  % looking at the data
  for i = 1:Ndata
    lege = [""];
    plotN++;
    figure(plotN)
    hold on
    for j= 1:8 
      plot(dataset.(strcat("data",(num2str(i)))).time,dataset.(strcat("data",(num2str(i)))).(strcat("channel",(num2str(j)))))
      lege = [lege; strcat("Channel ",(num2str(j)))];
    endfor

    plot(dataset.(strcat("data",(num2str(i)))).time,dataset.(strcat("data",(num2str(i)))).class*0.0001)
    lege = [lege; strcat("data ",(num2str(i)))];
    legend(lege);
    % hold off
  endfor
endif

%% Running PCA 
% [coeff,score,latent] = pca(dataset.data1.channel1) %not implemented yet
if(PCA) %% This mostly works.. but is there something better?
  if(plotAllPCvTime) % for each dataset, plot PC over time
    plotN++;
    lege = [""];
  endif


  for j = 1:Ndata
      if(any(dataset.(strcat("data",(num2str(j)))).class(:) == 7))
        j++;
      endif
    dat = [];
    for i = 1:8
      dat = [dat,dataset.(strcat("data",(num2str(j)))).(strcat("channel",(num2str(i))))];
    endfor
    men = mean(dat);
    cent = center(dat);
    covar = cov(cent);
    [W, eigs] = eig(covar);
    eigVals = diag(eigs);
    n = length(dat);
    Var = eigVals/(n-1);
    pc.(strcat("data",(num2str(j)))).data = (W*cent')';

    [maxVar, imaxVar] = max(Var./sum(Var));

    bPCA(:,1) = Var./sum(Var);
    bPCA(:,2) = 1:length(Var);
    bPCA = sortrows(bPCA,-1);
    pc.(strcat("data",(num2str(j)))).bPCA = bPCA;


    temp = W(:,imaxVar);
    for k = 1:3
      [tempV,tempI] = max(abs(temp));
      bChan(k) = tempI;
      temp(tempI) = 0;
    endfor
    printf("Best Channels for Dataset %d: ",j)
    printf("%d ", bChan)
    printf(" Data captured: %d\n", maxVar)

    %% Scale norm time
    ptime.(strcat("data",(num2str(j)))) = dataset.(strcat("data",(num2str(j)))).time/dataset.(strcat("data",(num2str(j)))).time(end);

    if(plotAllPCvTime) % for each dataset, plot PC over time
      hold on
      figure(plotN)
      plot(ptime.(strcat("data",(num2str(j)))),pc.(strcat("data",(num2str(j)))).data(:,imaxVar))
      lege = [lege; strcat("Data ",(num2str(j)))];
      legend(lege);
    endif
  endfor

  if(plotPCoriVt) % for D1 to D7, plot corrisponding rotated dataset in 3D over time
    plotN++;
    hold on
    figure(plotN)
    subplot(2,1,1)
    plot(ptime.data1,pc.data1.data(:,pc.data1.bPCA(1,2)))
    subplot(2,1,2)
    plot(ptime.data6,pc.data6.data(:,pc.data6.bPCA(1,2)))
    hold off

    plotN++;
    figure(plotN)
    hold on
    plot(ptime.data2,pc.data2.data(:,pc.data2.bPCA(1,2)))
    plot(ptime.data7,pc.data7.data(:,pc.data7.bPCA(1,2)))
    hold off

    plotN++;
    figure(plotN)
    hold on
    plot(ptime.data3,pc.data3.data(:,pc.data3.bPCA(1,2)))
    plot(ptime.data4,pc.data4.data(:,pc.data4.bPCA(1,2)))
    hold off

    plotN++;
    figure(plotN)
    hold on
    xx = ptime.data5(1:8680);
    yy = pc.data5.data(1:8680,pc.data5.bPCA(1,2));
    zz = pc.data5.data(1:8680,pc.data5.bPCA(2,2));
    aa = pc.data5.data(1:8680,pc.data5.bPCA(3,2));
    scatter3(xx,yy,zz,[],aa)
  endif

  if(plot3DD5chan)
    plotN++;
    figure(plotN)
    hold on
    xx = dataset.data5.time;
    yy = dataset.data5.channel6;
    zz = dataset.data5.channel5;
    aa = dataset.data5.channel2;
    scatter3(xx,yy,zz,[],aa)
  endif


  if(plotD2D7G2G3)
    plotN++;
    figure(plotN)
    hold on
    x2 = ptime.data2(1000:8000)-0.009;
    y2 = pc.data2.data(1000:8000,pc.data2.bPCA(1,2));
    z2 = pc.data2.data(1000:8000,pc.data2.bPCA(2,2));
    a2 = pc.data2.data(1000:8000,pc.data2.bPCA(3,2));
    scatter3(x2,y2,z2,[],a2)
    
    x7 = ptime.data7(1:8000);
    y7 = pc.data7.data(1:8000,pc.data7.bPCA(1,2));
    z7 = pc.data7.data(1:8000,pc.data7.bPCA(2,2));
    a7 = pc.data7.data(1:8000,pc.data7.bPCA(3,2));
    scatter3(x7,y7,z7,[],a7,"s","filled")
    hold off

    plotN++;
    figure(plotN)
    x2 = ptime.data2(1000:8000)-0.009;
    y2 = pc.data2.data(1000:8000,pc.data2.bPCA(1,2));
    z2 = pc.data2.data(1000:8000,pc.data2.bPCA(2,2));
    a2 = pc.data2.data(1000:8000,pc.data2.bPCA(3,2));
    subplot(2,1,1)
    scatter3(x2,y2,z2,[],'b')
    
    x7 = ptime.data7(1:8000);
    y7 = pc.data7.data(1:8000,pc.data7.bPCA(1,2));
    z7 = pc.data7.data(1:8000,pc.data7.bPCA(2,2));
    a7 = pc.data7.data(1:8000,pc.data7.bPCA(3,2));
    subplot(2,1,2)
    scatter3(x7,y7,z7,[],'r')

    plotN++;
    figure(plotN)
    hold on
    x2 = ptime.data2(8000:13500)-0.02;
    y2 = pc.data2.data(8000:13500,pc.data2.bPCA(1,2));
    z2 = pc.data2.data(8000:13500,pc.data2.bPCA(2,2));
    a2 = pc.data2.data(8000:13500,pc.data2.bPCA(3,2));
    scatter3(x2,y2,z2,[],'b')
    
    x7 = ptime.data7(7000:12500);
    y7 = pc.data7.data(7000:12500,pc.data7.bPCA(1,2));
    z7 = pc.data7.data(7000:12500,pc.data7.bPCA(2,2));
    a7 = pc.data7.data(7000:12500,pc.data7.bPCA(3,2));
    scatter3(x7,y7,z7,[],'r')
    hold off

    plotN++;
    figure(plotN)
    x2 = ptime.data2(8000:13500)-0.02;
    y2 = pc.data2.data(8000:13500,pc.data2.bPCA(1,2));
    z2 = pc.data2.data(8000:13500,pc.data2.bPCA(2,2));
    a2 = pc.data2.data(8000:13500,pc.data2.bPCA(3,2));
    subplot(2,1,1)
    scatter3(x2,y2,z2,[],'b')
    
    x7 = ptime.data7(7000:12500);
    y7 = pc.data7.data(7000:12500,pc.data7.bPCA(1,2));
    z7 = pc.data7.data(7000:12500,pc.data7.bPCA(2,2));
    a7 = pc.data7.data(7000:12500,pc.data7.bPCA(3,2));
    subplot(2,1,2)
    scatter3(x7,y7,z7,[],'r')
  endif


  %% Finding shapes
  % windowed shapes
  if(windowShape)
    winM = 1000;
    winS = 100;
    % for every win elements of data in time
    n = length(pc.data1.data);
    
    % mean
    mwinM = mod(n,winM);
    ewinM = n-mwinM-winM;
    for(i = 1:ewinM)
      sect = i:i+winM;
      datSect = pc.data1.data(sect);
      % find mean
      mu(sect) = mean(datSect);
    endfor
    sect = ewinM:n;
    datSect = pc.data1.data(sect);
    mu(sect) = mean(datSect);
    mu = mu';

    mwinS = mod(n,winS);
    ewinS = n-mwinS-winS;

    %% SD
    for(i = 1:ewinS)
      sect = i:i+winS;
      datSect = pc.data1.data(sect);
      % find variance;
      % split pos and neg to find variance
      % sig(i:i+win) = var(pc.data1.data(i:i+win));
      allPos = (datSect > 0) + (datSect == 0);
      allNeg = datSect < 0;

      posSig(sect) = sqrt(var(allPos.*datSect));
      negSig(sect) = -sqrt(var(allNeg.*datSect));
      sig(sect) = posSig(sect) + negSig(sect); 

    endfor
    sect = ewinS:n;
    datSect = pc.data1.data(sect);
    allPos = (datSect > 0) + (datSect == 0);
    allNeg = datSect < 0;
    posSig(sect) = sqrt(var(allPos.*datSect));
    negSig(sect) = sqrt(var(allNeg.*datSect));
    sig(sect) = posSig(sect) + negSig(sect); 


    % sig = sig';

    if(shapePlots)
      plotN++;
      figure(plotN)
      hold on 
      x = ptime.data1;
      y = pc.data1.data(:,pc.data1.bPCA(1,2));
      % scatter(x,y)
      plot(x,mu,"b")
      plot(x,posSig, "r")
      plot(x,negSig,"r")
      hold off
    endif
  endif

  if(windowPeaks)
    %pca1
    pca1 = pc.data1.data(:,pc.data1.bPCA(1,2));
    pc.data1.pksPosiPC1 = [];
    pc.data1.pksNegiPC1 = [];
    pksPosi1 = pc.data1.pksPosiPC1;
    pksNegi1 = pc.data1.pksNegiPC1;

    %pca2
    pca2 = pc.data1.data(:,pc.data1.bPCA(2,2));
    pc.data1.pksPosiPC2 = [];
    pc.data1.pksNegiPC2 = [];
    pksPosi2 = pc.data1.pksPosiPC2;
    pksNegi2 = pc.data1.pksNegiPC2;
    
    %time
    t = ptime.data1;
    dt = t(2)-t(1);

    % PCA1
    posPC1 =  ((pca1 > 0) + (pca1 == 0)).*pca1;
    [pksPos1,pksPosi1] = findpeaks(posPC1,"MinPeakDistance",round(0.01/dt));%,"MinPeakWidth",round(0.01/dt));

    negPC1 = -(pca1<0).*pca1;
    [pksNeg1,pksNegi1] = findpeaks(negPC1,"MinPeakDistance",round(0.01/dt));%,"MinPeakWidth",round(0.01/dt));

    pksPosi1 = pksPosi1(pksPos1(:)<0.001);
    pksNegi1 = pksNegi1(pksNeg1(:)>-0.001);

    % PCA2
    % posPC2 =  ((pca2 > 0) + (pca2 == 0)).*pca2;
    % [pksPos2,pksPos2i] = findpeaks(posPC2,"MinPeakDistance",round(0.009/dt));%,"MinPeakWidth",round(0.01/dt));

    % negPC2 = -(pca2<0).*pca2;
    % [pksNeg2,pksNeg2i] = findpeaks(negPC2,"MinPeakDistance",round(0.009/dt));%,"MinPeakWidth",round(0.01/dt));

    % pksPosi2 = pksPos2i(pksPos2(:)<0.001);
    % pksNegi2 = pksNeg2i(pksNeg2(:)>-0.001);

    plotsOn = false;

    if(plotsOn)
      plotN++;
      figure(plotN)
      hold on 
      % scatter(t,pca1);
      % plot(t,zppc1);
      % plot(t,znpc1);
      plot(t(pksPosi1),pca1(pksPosi1));
      plot(t(pksNegi1),pca1(pksNegi1));
      plot(t(pksPosi2),pca2(pksPosi2));
      plot(t(pksNegi2),pca2(pksNegi2));
      hold off

      plotN++;
      figure(plotN)
      hold on 
      scatter3(t(pksPosi1),pca1(pksPosi1),pca2(pksPosi1))
      scatter3(t(pksNegi1),pca1(pksNegi1),pca2(pksNegi1))
      scatter3(t,cirx(1:length(t)),ciry(1:length(t)))
      hold off

      zzPos = pca2(pksPosi1) + pca2(pksPosi1)'; 
      zzNeg = pca2(pksNegi1) + pca2(pksNegi1)'; 
      plotN++;
      figure(plotN)
      hold on 
      mesh(t(pksPosi1),pca1(pksPosi1),zzPos);
      mesh(t(pksNegi1),pca1(pksNegi1),zzNeg);
      hold off
    

      r = 0.0012;
      rad = 0:(2*pi)/length(t):2*pi;
      cirx = r.*cos(rad);
      ciry = r.*sin(rad);
      plotN++;
      figure(plotN)
      hold on 
      scatter(pca1(pksPosi1),pca2(pksPosi1))
      scatter(pca1(pksNegi1),pca2(pksNegi1))
      scatter(cirx,ciry)
      hold off
    endif

    rll = 500;
    movavg(posPC1(posPC1<0.001),rll,rll)

    rll = 5
    movavg(pca1(pksPosi1),rll,rll)
  endif
endif

%% Running LDA
if(LDA) %% Not currently working, pls seek help
  % for(i = 1:Ndata)
  %   dat.(num2str(i)).data = [];
  %   dat.(num2str(i)).time= [];
  %   dat.(num2str(i)).class= [];
  %   for(j = 1:8)
  %     dat.(num2str(i)).data = [dat.(num2str(i)).data,dataset.(strcat("data",(num2str(i)))).(strcat("channel",(num2str(j))))];
  %   endfor
  %   dat.(num2str(i)).time = dataset.(strcat("data",(num2str(i)))).time;
  %   dat.(num2str(i)).class = dataset.(strcat("data",(num2str(i)))).class;
  % endfor
  dat = [];
  for(i = 1:8)
    dat = [dat, dataset.data1.(strcat("channel",(num2str(i))))];
  endfor

  tempDat = dat(dataset.data1.class(:)==1,:);
  tempClass = [dataset.data1.class(dataset.data1.class()==1)]; %...
                % dataset.data1.class(dataset.data1.class()==1), ...
                % dataset.data1.class(dataset.data1.class()==1), ...
                % dataset.data1.class(dataset.data1.class()==1), ...
                % dataset.data1.class(dataset.data1.class()==1), ...
                % dataset.data1.class(dataset.data1.class()==1), ...
                % dataset.data1.class(dataset.data1.class()==1), ...
                % dataset.data1.class(dataset.data1.class()==1)];
  CC = train_sc(tempDat,tempClass,'LDA'); %% IDK how to format data for this function 
endif

%% Trying something
DPD = false;
if(DPD)
  plotOn = false;
  if(plotOn)
    dat = dataset.data1;

    figure(++plotN)
    subplot(3,2,1)
    scatter(dat.channel1(dat.class(:)==1),dat.channel2(dat.class(:)==1))
    subplot(3,2,2)
    scatter(dat.channel1(dat.class(:)==2),dat.channel2(dat.class(:)==2))
    subplot(3,2,3)
    scatter(dat.channel1(dat.class(:)==3),dat.channel2(dat.class(:)==3))
    subplot(3,2,4)
    scatter(dat.channel1(dat.class(:)==4),dat.channel2(dat.class(:)==4))
    subplot(3,2,5)
    scatter(dat.channel1(dat.class(:)==5),dat.channel2(dat.class(:)==5))
    subplot(3,2,6)
    scatter(dat.channel1(dat.class(:)==6),dat.channel2(dat.class(:)==6))

    figure(++plotN)
    scatter3(dataset.data1.channel1, dataset.data1.channel2, dataset.data1.class)
  endif

  %% Re-org data
  dat.data = [];
  dat.time = [];
  dat.class = [];
  dat.len = [];
  for(i = 1:Ndata);
    dat.len = [dat.len;length(dataset.(strcat("data",(num2str(i)))).channel1)];
    dat.time = [dat.time;dataset.(strcat("data",(num2str(i)))).time];
    dat.class = [dat.class;dataset.(strcat("data",(num2str(i)))).class];
    temp = [];
    for j = 1:NChan
      temp = [temp,dataset.(strcat("data",(num2str(i)))).(strcat("channel",(num2str(j))))];
    endfor
    dat.data = [dat.data; temp];
  endfor

  %% Length of each guesture per length of each dataset
  temp = [];
  temp(1) = 0;
  for(i = 2:Ndata+1)
    temp(i) = sum(dat.len(1:i-1));
  endfor
  temp(1) = 1;
  DPD.len = [];
  for(i = 1:Ndata)
    for(j = 1:NG)
      DPD.len(i,j) = length(dat.class(dat.class(temp(i):temp(i+1))==j));
    endfor
  endfor
  % DPD.G1.len(3) = length(dat.class(dat.class(sum(dat.len(1:2)):sum(dat.len(1:3)))==1));

  %% Distance of each point to the center (0) for each guesture in 8 channel dimensions
  DPD.Dist = [];

  for(k = 1:NChan)
    for(j = 1:NChan)
      temp = [];
      for(i = 1:NG)
        temp = [temp;sqrt(sum(dat.data(dat.class(:) == i,k:j).^2,2))];
      endfor
      DPD.Dist = [DPD.Dist, temp];
    endfor
  endfor
  temp = [];
  for(i = 1:NG)
    temp = [temp; dat.class(dat.class(:)==i)];
  endfor
  DPD.Dist = [DPD.Dist, temp];

  %% Average and variation of distances per guesture
  temp1 = [];
  temp1(1) = 0;
  for(i = 2:NG+1)
    temp1 = [temp1; DPD.len(:,i-1)];
  endfor
  temp = [];
  temp(1) = 0;
  for(i = 2:length(temp1))
    temp(i) = sum(temp1(1:i));
  endfor
  temp = temp';
  temp(1) = 1;

  DPD.mu = [];
  DPD.sig = [];
  DPD.class = [];

  for(j = 1:size(DPD.Dist,2)-1)
    tempmu = [];
    tempsig = [];
    for(i = 1:length(temp)-1)
      tempmu = [tempmu;mean(DPD.Dist(temp(i):temp(i+1),j))];
      tempsig = [tempsig;var(DPD.Dist(temp(i):temp(i+1),j))];
      DPD.class(i) = DPD.Dist(temp(i),end);
    endfor
    DPD.mu = [DPD.mu, tempmu];
    DPD.sig = [DPD.sig, tempmu];
  endfor

  % for(i = 1:Ndata*NG)
  %   for(j = 1:size(DPD.Dist,2)-1)
  %     DPD.mu = [DPD.mu;mean(DPD.Dist(temp(i):temp(i+1),j))];
  %     DPD.sig = [DPD.sig;var(DPD.Dist(temp(i):temp(i+1),j))];
  %     DPD.class = [DPD.class;DPD.Dist(temp(i),end)];
  %   endfor
  % endfor
  DPD.mu2 = [];
  DPD.sig2 = [];
  % for(i = 1:)
    DPD.mu2 = sqrt(sum(DPD.mu.^2,2));
    DPD.sig2 = sqrt(sum(DPD.sig.^2,2));
  % endfor

  plotN++;
  numplts = 6;
  for(i = 1:numplts)
    figure(plotN)
    subplot(floor(sqrt(numplts)), ceil(sqrt(numplts)),i)
    scatter3(DPD.mu(:,i),DPD.mu(:,i+1),DPD.mu(:,i+2),50*DPD.class,DPD.class,"filled");
    figure(plotN+1)
    subplot(floor(sqrt(numplts)), ceil(sqrt(numplts)),i)
    scatter3(DPD.sig(:,i),DPD.sig(:,i+1),DPD.sig(:,i+2),50*DPD.class,DPD.class,"filled");
    figure(plotN+2)
    subplot(floor(sqrt(numplts)), ceil(sqrt(numplts)),i)
    scatter3(DPD.sig(:,i),DPD.mu(:,i),DPD.mu(:,i+1),50*DPD.class,DPD.class,"filled");
    figure(plotN+3)
    subplot(floor(sqrt(numplts)), ceil(sqrt(numplts)),i)
    scatter3(DPD.mu(:,i),DPD.sig(:,i),DPD.sig(:,i+1),50*DPD.class,DPD.class,"filled");
  endfor
  plotN += 3;
endif

%% fft 
FFT = true;
if(FFT) %% Need help with movfun
  dat = dataset.data1.channel1;
  dat = dat';
  rll = 250;
  movfun(@fft, dat, rll,"Endpoints","shrink"); %% Also not working
endif


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% y1 = [9,0,5,4,3,0,9,8]
% x1 = [1,2,3,4,5,6,7,8]

% y2 = [5,6,4,6,8,4,5,3,1,6,8,8,9,4]
% x2 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]

% hold on
% figure(1)
% plot(x1,y1)
% plot(x2,y2)



% combMag = sqrt(sum(dat.^2,2));
% figure(2)
% plot(dataset.data1.time,combMag)


%% FFT
% Fs = length(dataset.data1.channel1)/(dataset.data1.time(end)/1000) %sampling frequency
% T = 1/Fs;
% L = dataset.data1.time(end);

% Y = fft(dataset.data1.channel1);
% P2 = abs(Y/L);
% P1 = P2(1:L/2+1);
% P1(2:end-1) = 2*P1(2:end-1);

% f = Fs*(0:(L/2))/L;
% figure()
% hold on
% plot(f,P1)
% title('Single-Sided Amplitude Spectrum of Channel')
% xlabel('f (Hz)')
% ylabel('|P1(f)|')

% Fs = length(dataset.data1.channel2)*1000/dataset.data1.time(end) %sampling frequency
% T = 1/Fs;
% L = dataset.data1.time(end);

% Y = fft(dataset.data1.channel2);
% P2 = abs(Y/L);
% P1 = P2(1:L/2+1);
% P1(2:end-1) = 2*P1(2:end-1);

% f = Fs*(0:(L/2))/L;

% plot(f,P1)
