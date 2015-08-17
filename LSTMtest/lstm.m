%sequence data to learn
X = [1 0 1 1 0 1 1 1 0 1 1 1 1 0 1 0 1 1 0 1 1 1 0 1 1 1 1 0 1 0 1 1 0 1 1 1 0 1 1 1 1]';
%store last time step's output
ht_last = 0;
%cell state form last time step
ct_last = 0;
m = size(X, 1);
% weight matrices
theta_xi = 2 * randn(2,1); %weights for input X and bias to input gate
theta_xf = 2 * randn(2,1); %weights for input X and bias to forget gate
theta_xo = 2 * randn(2,1); %weights for input X and bias to output gate
theta_xg = 2 * randn(2,1); %weights for input X and bias to net input
theta_hi = 2 * randn(1,1); %weights for output h to input gate
theta_hf = 2 * randn(1,1); %weights for output h to forget gate
theta_ho = 2 * randn(1,1); %weights for output h to output gate
theta_hg = 2 * randn(1,1); %weights for output h to net input

thetaVec = [theta_xi; theta_xf; theta_xo; theta_xg; theta_hi; theta_hf; theta_ho; theta_hg];
%  value to see how more training helps.
options = optimoptions(@fminunc, 'MaxIter', 5000, 'Display', ...
'Iter', 'TolFun', 1e-9, 'TolFun', 1e-12, 'MaxFunEvals', 2000);

% Create "short hand" for the cost function to be minimized
costFunction = @(p) costLSTM(X, thetaVec);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[opt_theta, cost] = fminunc(costFunction, thetaVec, options);

theta_xi = reshape(opt_theta(1:2), 2, 1); %weights for input X and bias to input gate
theta_xf = reshape(opt_theta(3:4), 2, 1); %weights for input X and bias to forget gate
theta_xo = reshape(opt_theta(5:6), 2, 1); %weights for input X and bias to output gate
theta_xg = reshape(opt_theta(7:8), 2, 1); %weights for input X and bias to net input
theta_hi = reshape(opt_theta(9), 1, 1); %weights for output h to input gate
theta_hf = reshape(opt_theta(10), 1, 1); %weights for output h to forget gate
theta_ho = reshape(opt_theta(11), 1, 1); %weights for output h to output gate
theta_hg = reshape(opt_theta(12), 1, 1); %weights for output h to net input

results = [];
for i = 1:m
    x = X(i);
    inputs = [x; 1]; %add input, bias
    
    input_gate_input = (theta_xi' * inputs + theta_hi * ht_last);
    input_gate_output = sigmoid(input_gate_input);
    
    g_input = (theta_xg' * inputs + theta_hg * ht_last);
    g_output = tanh(g_input);
    
    output_gate_input = (theta_xo' * inputs + theta_ho * ht_last);
    output_gate_output = sigmoid(output_gate_input);
    
    forget_gate_input = (theta_xf' * inputs + theta_hf * ht_last);
    forget_gate_output = sigmoid(forget_gate_input);
    
    ct = forget_gate_output + ct_last  + (input_gate_output .* g_output);
    ct_last = ct;
    
    ht = output_gate_output .* tanh(ct); %final output
    ht_last = ht;
    results(i) = ht;
end

disp(round(results));
