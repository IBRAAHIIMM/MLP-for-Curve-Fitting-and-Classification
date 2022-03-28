    %Assignment Question 1 Part B
clc 
clear
fprintf("-----Gradient Descent-----\n");
learning_rate = 1.0;
weights = 2*rand(2,1) -1;
w = weights; 
fprintf("Randomly initialized (x,y) = (%0.3f,%0.3f)\n",weights(1,1),weights(2,1));
error = f(weights)-0;
iteration = 1;
function_out = f(weights(:,iteration));

while error >0.0001
    iteration = iteration +1;
    function_out(:,iteration) = f(weights(:,iteration-1));
    weights(:,iteration) = weights(:,iteration-1)-learning_rate*gradient(weights(:,iteration-1));
    error = f(weights(:,iteration))-0;
end
fprintf("Learning Rate = %0.5f\n",learning_rate);
fprintf("Minimum value for gradient descent is achieved in %d iterations with the error value %0.7f\n",iteration,error);
fprintf("Final (x,y) = (%f,%0.5f)\n",weights(1,size(weights,2)),weights(2,size(weights,2)));
figure(1)
subplot(2,1,1);
plot(weights(1,:),weights(2,:));
xlabel('x');
ylabel('y');
title(['x and y trajectory when learning rate is ',num2str(learning_rate)])

n = 1:1:iteration;
subplot(2,1,2);
plot(n,function_out);
xlabel('Iteration');
ylabel('function value');
title('Function value versus iteration')
saveas(gcf,"Q1partB.jpg");

%Question 1 Part C

fprintf("----Newton's Method-----\n");
weights_c = w;
error = f(weights_c)-0;
iteration_c = 1;
function_out_c = f(weights_c(:,iteration_c));

while error >0.0001
    iteration_c = iteration_c +1;
    function_out_c(:,iteration_c) = f(weights_c(:,iteration_c-1));
    weights_c(:,iteration_c) = weights_c(:,iteration_c-1)-inv(hessian(weights_c(:,iteration_c-1)))*gradient(weights_c(:,iteration_c-1));
    error = f(weights_c(:,iteration_c))-0;
end
fprintf("Minimum value for Newton's method is achieved in %d iterations with the error value %0.7f\n",iteration_c,error);
fprintf("Final (x,y) = (%0.5f,%0.5f)\n",weights_c(1,size(weights_c,2)),weights_c(2,size(weights_c,2)));

figure(2) 
subplot(2,1,1);
plot(weights_c(1,:),weights_c(2,:));
xlabel('x');
ylabel('y');
title('x and y trajectory for Newtons Method')

n = 1:1:iteration_c;
subplot(2,1,2);
plot(n,function_out_c);
xlabel('Iteration');
ylabel('function value');
title('Function value versus iteration')
saveas(gcf,"Q1partC.jpg");
function value = f(weights)
    x = weights(1);
    y =weights(2);
    value = (1-x)^2+100*(y-x^2)^2 ;   
end
function gk = gradient(weights)
    xk = weights(1);
    yk =weights(2);
    gxk = -2*(1-xk)-400*xk*(yk-xk^2);
    gyk = 200*(yk-xk^2);
   	gk = [gxk;gyk];
end

function hes= hessian(weights)
    x = weights(1);
    y = weights(2);
    hes = zeros(2,2);
    hes(1,1) = 2-400*y+1200*x^2;
    hes(1,2) =-400*x;
    hes(2,1) =-400*x;
    hes(2,2) =200;
end


