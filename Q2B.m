%Q2 new try
tic
x_train = -1.6:0.05:1.6;
y_train = f(x_train);
figure(1)
plot(x_train,y_train,"r.");
xlabel("x");
ylabel("y");
title("Training Set");

layers = [1:10 20 50];
epochs  =1000;

%net.layers{1}.transferFcn = 'tansig';
%net.layers{2}.transferFcn = 'purelin';
x_test = -1.6:0.01:1.6;
k = 1;
for n= layers
    net = feedforwardnet(n);
    net = configure(net, x_train, y_train); 
    net.trainParam.epochs = 5000;
    net.trainFcn = 'trainlm';
    net.performFcn = "mae";
    net.trainparam.lr = 0.001;
    fprintf("LayerNum: %2d ",n);
    idx = randperm(size(x_train,2));
    [net,a,e,pf]  = train(net, x_train(:,idx), y_train(:,idx));
    figure(k+1)
    subplot(2,1,1)
    plot(x_test,net(x_test),"r.",x_test,f(x_test),"-b");
    xlabel("x");
    ylabel("y");
    title("Batch Mode Training with trainlm and "+n+" neurons"); 
    legend(["Network Output","Expected Output"]);
    subplot(2,1,2)
    x_test2 = -3:0.01:3;
    plot(x_test2,net(x_test2),"r.",x_test2,f(x_test2),"-b");
    xlabel("x");
    ylabel("y");
    title("Batch Mode Training with trainlm and "+n+" neurons from -3 to 3"); 
    legend(["Network Output","Expected Output"]);
    k= 1+k;
    saveas(gcf, ['Batch mode number of neurons ',num2str(n),' with trainlm'], 'png');
end
toc

function y = f(x)
    y =1.2*sin(pi*x)-cos(2.4*pi*x);
end