%Q2 new try
tic 
x_train = -1.6:0.05:1.6;
y_train = f(x_train);
figure(1)
plot(x_train,y_train,"r.");
xlabel("x");
ylabel("y");
title("Training Set");

x_c = num2cell(x_train, 1);
y_c = num2cell(y_train, 1);
layers = [1:10 20 50];
epochs  =1000;

%net.layers{1}.transferFcn = 'tansig';
%net.layers{2}.transferFcn = 'purelin';
x_test = -1.6:0.01:1.6;
k = 1;
for n= layers
    net = feedforwardnet(n);
    net.inputWeights{2,1}.learnParam.lr = 0.01;
    net.biases{2,1}.learnParam.lr = 0.01;
    net = configure(net, x_c, y_c); 
    for i = 1:epochs
        fprintf("LayerNum: %2d Epochs: %4d\n",n,i)
        idx = randperm(size(x_c,2));
        net.trainFcn = 'traingd';
        net.performFcn = "mae";
        [net,a,e,pf]  = adapt(net, x_c(:,idx), y_c(:,idx));
    end 
    figure(k+1)
    subplot(2,1,1)
    plot(x_test,net(x_test),"r.",x_test,f(x_test),"-b");
    xlabel("x");
    ylabel("y");
    title("Sequential Mode Training with traingd and "+n+" neurons"); 
    legend(["Network Output","Expected Output"]);
    subplot(2,1,2)
    x_test2 = -3:0.01:3;
    plot(x_test2,net(x_test2),"r.",x_test2,f(x_test2),"-b");
    xlabel("x");
    ylabel("y");
    title("Sequential Mode Training with traingd and "+n+" neurons from -3 to 3"); 
    legend(["Network Output","Expected Output"]);
    k= 1+k;
    name = ['Sequential mode number of neurons ', num2str(n), ' traingd'];
    saveas(gcf,name,'jpg');
end

toc
function y = f(x)
    y =1.2*sin(pi*x)-cos(2.4*pi*x);
end

