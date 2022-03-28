load("Q3A_data.mat",'train_set_inp','train_set_out','test_set_inp','test_set_out');
neurons = 20;
%net.trainParam.epochs = 1000;
%net.performFcn = "mse";
%net.layers{1}.transferFcn = 'tansig';
%net.layers{2}.transferFcn = 'hardlims';
%net.trainparam.lr = 0.001;
%     compet - Competitive transfer function.
%     elliotsig - Elliot sigmoid transfer function.
%     hardlim - Positive hard limit transfer function.
%     hardlims - Symmetric hard limit transfer function.
%     logsig - Logarithmic sigmoid transfer function.
%     netinv - Inverse transfer function.
%     poslin - Positive linear transfer function.
%     purelin - Linear transfer function.
%     radbas - Radial basis transfer function.
%     radbasn - Radial basis normalized transfer function.
%     satlin - Positive saturating linear transfer function.
%     satlins - Symmetric saturating linear transfer function.
%     softmax - Soft max transfer function.
%     tansig - Symmetric sigmoid transfer function.
%     tribas - Triangular basis transfer function.

for n = neurons
    net = patternnet(n);
    net = configure(net, train_set_inp, train_set_out);
    net.trainFcn = 'traingdm';
    [net,tr]  = train(net, train_set_inp, train_set_out);
    bins = [0,1];
    countTrain = hist(net(train_set_inp)>= 0.5==train_set_out,bins); %output layer activation is logsig therefore threshold is 0.5
    acc_tran = countTrain(2)/size(train_set_inp,2);
    countTest= hist(net(test_set_inp)>= 0.5 ==test_set_out,bins);
    acc_test  = countTest(2)/size(test_set_inp,2);
    fprintf("Accuracy of MLP network with %d neurons for training data: %0.2f percent\n",n,acc_tran*100);
    fprintf("Accuracy of MLP network with %d neurons for test data: %0.2f percent\n",n,acc_test*100);
end
save("Q3C_data.mat");