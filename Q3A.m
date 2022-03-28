%Question 3 Part A
deerDirect = "C:\Users\ibrahim\Desktop\EEE3.2\EE5904\HW2\group_2\deer\";
shipDirect = "C:\Users\ibrahim\Desktop\EEE3.2\EE5904\HW2\group_2\ship\";
deers = zeros(32*32,500);
ship = zeros(32*32,500);
for i = 0:499
    selectPic = [num2str(rem(fix(i/100),10)),num2str(rem(fix(i/10),10)),num2str(rem(i,10)),'.jpg'];
    deerPictureDirect = deerDirect+selectPic;
    shipPictureDirect = shipDirect+selectPic;
    deerPic = imread(deerPictureDirect);
    deerPic = reshape(rgb2gray(deerPic),[32*32,1]);
    deers(:,i+1) = deerPic;
    shipPic = imread(shipPictureDirect);
    shipPic = reshape(rgb2gray(shipPic),[32*32,1]);
    ship(:,i+1) = shipPic;
end
deerTrain_input = deers(:,1:450);
deerTrain_out = ones(1,450);
deerTest_input = deers(:,451:500);
deerTest_out = ones(1,50);
shipTrain_input = ship(:,1:450);
shipTrain_out = zeros(1,450);
shipTest_input = ship(:,451:500);
shipTest_out =  zeros(1,50);

train_set_inp = [deerTrain_input,shipTrain_input];
train_set_out = [deerTrain_out,shipTrain_out];
test_set_inp = [deerTest_input,shipTest_input];
test_set_out  = [deerTest_out,shipTest_out];
net = perceptron;
idx = randperm(size(train_set_inp,2));
[net, tr] = train(net,train_set_inp(:,idx) ,train_set_out(:,idx));

bins = [0,1];
countTrain = hist(net(train_set_inp)==train_set_out,bins);
acc_tran = countTrain(2)/size(train_set_inp,2);
countTest= hist(net(test_set_inp)==test_set_out,bins);
acc_test  = countTest(2)/size(test_set_inp,2);
fprintf("Accuracy of trained network for training data: %0.2f %\n",acc_tran*100);
fprintf("Accuracy of trained network for test data: %0.2f %\n",acc_test*100);
save("Q3A_data.mat");


