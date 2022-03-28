clear all
clc
load("Q3A_data.mat",'train_set_inp','train_set_out','test_set_inp','test_set_out');
Ts_mean = mean(train_set_inp,'all'); 
repmean_train(1:size(train_set_inp,1),1:size(train_set_inp,2)) = Ts_mean;
Ts_var = var(train_set_inp,0,'all');
newTs_inp = (train_set_inp-repmean_train)./Ts_var.*10;
TestS_mean = mean(test_set_inp,'all');
repmean_test(1:size(test_set_inp,1),1:size(test_set_inp,2)) = TestS_mean;
TestS_var = var(test_set_inp,0,'all');
newTestS_inp = (test_set_inp-repmean_test)./TestS_var.*10;

net = perceptron;
idx = randperm(size(newTs_inp,2));
[net, tr] = train(net,newTs_inp(:,idx) ,train_set_out(:,idx));

bins = [0,1];
countTrain = hist(net(newTs_inp)==train_set_out,bins);
acc_tran = countTrain(2)/size(train_set_inp,2);
countTest= hist(net(newTestS_inp)==test_set_out,bins);
acc_test  = countTest(2)/size(test_set_inp,2);
fprintf("Accuracy of trained network with normalized inputs for training data: %0.2f percent\n",acc_tran*100);
fprintf("Accuracy of trained network with normalized inputs for test data: %0.2f percent\n",acc_test*100);
save("Q3B_data.mat");