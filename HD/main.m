clear all;

% [Parameter Initialization]
max_iter = 2;	% maximum iteration number
O = 3;	% dimension of the observations
Q = 3;  % number of states
p1 = 1; % lower limits of the RF-fragment's truncation window
p2 = 1; % upper limits of the RF-fragment's truncation window
max_limits = 2; % auxiliary parameters
W = 6;	% the size of RF-fragment
act_fun = 'tansig';             % activation function of DNN 
loss_fun = 'crossentropy';      % loss function of DNN
hidden_layers=[24 24 24 24];    % structure of DNN

% [train HMM-DNN]
[train_samples, train_labels, test_samples, test_labels] = SplitDataset(); % split the dataset
[model, net, percent, accuracy] = TrainHMMDNN(train_samples, train_labels, O, Q, W, max_limits, p1, p2, hidden_layers, max_iter, loss_fun, act_fun);

% [train rear classifier]
[classifier_net, classifier_accuracy, con_matrix] = TrainRearClassifier(train_samples, train_labels, W, max_limits, p1, p2, percent, model, net);


% [test]
test_data = {};
for j = 1:length(test_samples)
    test_data{j} = test_samples{j}(1:O, :);
end
err_con = Testing(test_data, test_labels, W, max_limits, p1, p2, percent, model, net, classifier_net)
F1Score(err_con(2, 2), err_con(1, 2), err_con(2, 1));
kappa(err_con(1, 1), err_con(1, 2), err_con(2, 1), err_con(2, 2));
























