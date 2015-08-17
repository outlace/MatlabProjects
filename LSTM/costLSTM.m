function J = costLSTM(X, thetaVec)
J = 0;

%store last time step's output
ht_last = 0;
%cell state form last time step
ct_last = 0;
m = size(X, 1);
% weight matrices
theta_xi = reshape(thetaVec(1:2), 2, 1); %weights for input X and bias to input gate
theta_xf = reshape(thetaVec(3:4), 2, 1); %weights for input X and bias to forget gate
theta_xo = reshape(thetaVec(5:6), 2, 1); %weights for input X and bias to output gate
theta_xg = reshape(thetaVec(7:8), 2, 1); %weights for input X and bias to net input
theta_hi = reshape(thetaVec(9), 1, 1); %weights for output h to input gate
theta_hf = reshape(thetaVec(10), 1, 1); %weights for output h to forget gate
theta_ho = reshape(thetaVec(11), 1, 1); %weights for output h to output gate
theta_hg = reshape(thetaVec(12), 1, 1); %weights for output h to net input

m = size(X, 1);
results = [];
for i = 1:(m - 1)
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

for n = 1:(m-1)
    a3n = results(n)';
    yn = X(n + 1,:)';
    J = J + ( -yn'*log(a3n) - (1-yn)'*log(1-a3n) );
end
J = (1/m) * J;


end