clear all;
close all;
TITLE={'$I_{\rm up}$ (A)', '$I_{\rm mid}$ (A)', '$I_{\rm down}$ (A)', '$I_{\rm tot}$ (A)',...
    '$T_{\rm max}$ ($^\circ \rm C$)', '$T_{\rm min}$ ($^\circ \rm C$)',...
    '$T_{\rm up}$ ($^\circ \rm C$)', '$T_{\rm mid}$ ($^\circ \rm C$)', '$T_{\rm down}$ ($^\circ \rm C$)',...
    'SU (1)'};
NEURONS=[10,10,10,0,5,5,0,0,0];% I_up use 5 neurons, I_mid use 10 neurons, I_tot use 8 neurons,
EPOCH_TOTAL=3000; % In total, we train the network 1000 epochs (an epoch means that all of the training data is provided to the network once)
figure % draw the error curves during the training process
ax1=subplot(1,2,1);
ax2=subplot(1,2,2);
hold(ax1,'on');
hold(ax2,'on');
grid(ax1,'on');
grid(ax2,'on');
for i_output=[5 6 7 11 12]
    data=xlsread('SOEC_whole_cell_data_for_upload.xlsx','total_required_air_appended');% read data from a xls or xlsx file
    data=data(:,1:16);
    rng(114514);
    data=data(randperm(size(data,1)),:); % The original data are in sequence, which is bad for neural network. Therefore we disrupt the data sequence first.
    train_set=data(1:1500,:);
    test_set=data(1501:end,:);% There are 1200 data points in total 
    
    train_set_X=train_set(:,1:4);% get the inputs of the training set (Column 1-4)
    train_set_Y=train_set(:,i_output);% get the output of the training set (Column 5)
    test_set_X=test_set(:,1:4);% get the inputs of the test set (Column 1-4)
    test_set_Y=test_set(:,i_output);% get the output of the test set (Column 5)
    
    X_MAX=[750,240,300,1.7]; % get the maximum value of each input (each column)
    X_MIN=[600,60,40,1.0]; % get the minimum value of each input (each column)
    X_BASE=[675,100,170,1.35]; % used in plot the prediction surface of the NNs
    
    for i=1:size(train_set,1)
        train_set_X(i,:)=(train_set_X(i,:)-X_MIN)./(X_MAX-X_MIN)*2-1; % transform the input range into [-1,1]
    end
    for i=1:size(test_set,1)
        test_set_X(i,:)=(test_set_X(i,:)-X_MIN)./(X_MAX-X_MIN)*2-1; % transform the input range into [-1,1]
    end
    
    trainFcn = 'trainlm'; % set the training algorithm (the algorithm for adjusting the parameter of the neural network)
    hiddenLayerSize = [NEURONS(find([5       6        7      8      11      12 13     14      15     16]==i_output))]; % set the structure of the neural network (5 neurons in the hidden layer)
    net=fitnet(hiddenLayerSize,trainFcn); % create a neural network (an initial one that has not been trained)
    
    EPOCH_STEP=EPOCH_TOTAL/100; % In each step, we train the network 10 epochs. So there are 1000/10=100 steps in total
    net.trainParam.epochs=EPOCH_STEP; % In each step, we train the network 10 epochs
    
    net.trainParam.max_fail=1000; % stopping criteria
    net.trainParam.min_grad=1e-15;% stopping criteria
    net.divideParam.trainRatio = 100/100; % we use 100% of the training data for training the network
    net.divideParam.valRatio = 0/100;% we use 0% of the training data for validating the accuracy
    net.divideParam.testRatio = 0/100;% we use 0% of the training data for testing the accuracy
    net.trainParam.showWindow=false; % do not show the window of the training process
    
    errTrain=[]; % create an array to store the error of the neural network during the training process
    errTest=[]; % create an array to store the error of the neural network during the training process
    errTest_Max=[];
    errTrain_Max=[];
    for Step=1:EPOCH_TOTAL/EPOCH_STEP % In each step, we train the network, and then calculate the errors of the network
        Step
        [net,tr] = train(net,train_set_X',train_set_Y'); % train the network using the training set
        errTrain=[errTrain,norm(train_set_Y'-net(train_set_X'),2)/sqrt(length(train_set_X))];% calculate the error on the training set
        errTest=[errTest,norm(test_set_Y'-net(test_set_X'),2)/sqrt(length(test_set_X))];% calculate the error on the test set
    end
    
    if i_output<=7
        yyaxis(ax1,'left');
        plot(ax1,[EPOCH_STEP:EPOCH_STEP:EPOCH_TOTAL],errTrain,'LineWidth',2);
        yyaxis(ax2,'left');
        plot(ax2,[EPOCH_STEP:EPOCH_STEP:EPOCH_TOTAL],errTest,'LineWidth',2);
    else
        yyaxis(ax1,'right');
        plot(ax1,[EPOCH_STEP:EPOCH_STEP:EPOCH_TOTAL],errTrain,'LineWidth',2);
        yyaxis(ax2,'right');
        plot(ax2,[EPOCH_STEP:EPOCH_STEP:EPOCH_TOTAL],errTest,'LineWidth',2);
    end
    NET{find([5       6        7      8      11      12 13     14      15     16]==i_output)}=net;
end
L1=legend(ax1,'train-RMSE-$I_{\rm up}$','train-RMSE-$I_{\rm mid}$','train-RMSE-$I_{\rm down}$','train-RMSE-$T_{\rm max}$','train-RMSE-$T_{\rm min}$');
L2=legend(ax2,'test-RMSE-$I_{\rm up}$','test-RMSE-$I_{\rm mid}$','test-RMSE-$I_{\rm down}$','test-RMSE-$T_{\rm max}$','test-RMSE-$T_{\rm min}$');
xlabel(ax1,'Number of epochs','Interpreter','Latex');
xlabel(ax2,'Number of epochs','Interpreter','Latex');
yyaxis(ax1,'left');
ylabel(ax1,'Error (A)','Interpreter','Latex');
yyaxis(ax1,'right');
ylabel(ax1,'Error ($^\circ {\rm C}$)','Interpreter','Latex');
yyaxis(ax2,'left');
% ylim(ax2,[0,0.15]);
ylabel(ax2,'Error (A)','Interpreter','Latex');
yyaxis(ax2,'right');
ylabel(ax2,'Error ($^\circ {\rm C}$)','Interpreter','Latex');
set(L1,'Interpreter','Latex');
set(L2,'Interpreter','Latex');
set(ax1,'FontSize',12,'FontName','Times New Roman'); % set the font and fontsize
set(ax2,'FontSize',12,'FontName','Times New Roman'); % set the font and fontsize

save('surrogate_model_for_upload.mat','NET')
