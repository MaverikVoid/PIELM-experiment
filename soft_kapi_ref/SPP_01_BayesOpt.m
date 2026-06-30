clc; clear; close all;
rng(42);

%% ===================== Bayesian Optimization Setup =====================
nu = 1e-4;                 % PDE parameter
N  = 1000;                 % points per partition
kSigma = 5.0;              % RBF width scaling

w_var = optimizableVariable('w',[0.90,0.99]);

Objective = @(T) KAPIELM_Objective(T.w,nu,N,kSigma);

results = bayesopt(Objective, w_var, ...
    'MaxObjectiveEvaluations', 25, ...
    'AcquisitionFunctionName','expected-improvement-plus', ...
    'PlotFcn',{}, ...                 % Disable BO internal plotting
    'Verbose', 1);

bestW = results.XAtMinObjective.w;
bestLoss = results.MinObjective;

fprintf('Optimal w = %.5f,  Best J = %.3e\n', bestW, bestLoss);

%% ===================== Extract BO history for custom plotting =====================
allW    = results.XTrace.w;
allLoss = results.ObjectiveTrace;

% bestSoFar = min(cummin(allLoss));
bestSoFar = cummin(allLoss);

%% ===================== Publication-quality plot =====================
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultTextInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
set(groot,'defaultAxesFontSize',22);
set(groot,'defaultTextFontSize',24);

fig = figure('Color','w','Position',[100 100 2000 600]);

%% === Plot 1: BO Samples (objective vs w) ===
scatter(allW, allLoss, 60, 'MarkerFaceColor',[0.4 0.6 1], ...
    'MarkerEdgeColor','k', 'LineWidth',0.6);
hold on;

% Mark best w with a star
[bestLoss, idxBest] = min(allLoss);
plot(allW(idxBest), bestLoss, 'p', ...
    'MarkerSize', 18, 'MarkerFaceColor','r','MarkerEdgeColor','k');

xlabel('$w$', 'Interpreter','latex');
ylabel('$J(w)$', 'Interpreter','latex');
title('Bayesian Optimization Samples','Interpreter','latex');
grid on; box on;

exportgraphics(gcf,'TC_BO_Plots_Publication.png','Resolution',300);


%% =====================================================================
%% ====================== OBJECTIVE FUNCTION ============================
%% =====================================================================
function J = KAPIELM_Objective(w,nu,N,kSigma)

    xL=0; xR=1;

    % Partition lens
    lens = [w, 1-w];

    % Centers + widths
    [x_part, sigma_part, x_global, sigma_global, edges] = ...
        SAMP_POINTS_SIGMA(N, lens, kSigma);

    % Combine
    alpha_star = [x_part; x_global];
    sig_x      = [sigma_part; sigma_global];
    NN         = numel(alpha_star);

    % PDE collocation
    X_pde = sort(alpha_star);
    Nc    = NN;

    % RBF parameters
    m     = 1 ./ (sqrt(2) * sig_x);
    alpha = -m .* alpha_star;

    % PDE operator (u_x - nu u_xx = 0)
    z   = X_pde * m.' + alpha.'; 
    z2  = min(z.^2,700);
    phi = exp(-z2);

    phi_x  = -2 .* (ones(Nc,1)*m.') .* z .* phi;
    phi_xx =  2 .* (ones(Nc,1)*(m.'.^2)) .* (2*z2 - 1) .* phi;

    LHS_PDE = phi_x - nu .* phi_xx;
    RHS_PDE = zeros(Nc,1);

    % Boundary conditions
    X_bc   = [xL; xR];
    z_bc   = X_bc * m.' + alpha.';
    phi_bc = exp(-z_bc.^2);

    LHS_BC = phi_bc;
    RHS_BC = [0;1];

    % Assemble
    H = [LHS_PDE; LHS_BC];
    b = [RHS_PDE; RHS_BC];

    % Solve
    c = H\b;

    % Objective
    J = norm(H*c - b, Inf);

    % Penalize NaN / Inf in BO
    if isnan(J) || isinf(J)
        J = 1e6;
    end
end

%% =====================================================================
%% ====================== SUPPORTING FUNCTIONS ==========================
%% =====================================================================
function [x_part, sigma_part, x_global, sigma_global, edges] = SAMP_POINTS_SIGMA(N, customLens, kSigma)
    if nargin < 3 || isempty(kSigma), kSigma = 1.0; end
    customLens = customLens(:).';  k = numel(customLens);
    s = sum(customLens); 
    if abs(s-1) > 1e-12, customLens = customLens / s; end

    edges = [0, cumsum(customLens)]; edges(end)=1;

    % Partition centers
    x_part = [];
    for j = 1:k
        a = edges(j); b = edges(j+1);
        if j == 1 && j == k
            pts = linspace(a,b,N);
        elseif j == 1
            pts = linspace(a,b,N+1); pts = pts(1:N);
        elseif j == k
            pts = linspace(a,b,N+1); pts = pts(2:end);
        else
            pts = linspace(a,b,N+2); pts = pts(2:end-1);
        end
        x_part = [x_part, pts];
    end

    % Partition widths
    sigma_part = zeros(1, k*N);
    idx=0;
    for j=1:k
        sig_j = kSigma*(customLens(j)/N);
        sigma_part(idx+(1:N)) = sig_j;
        idx = idx + N;
    end

    % Global points
    xg = linspace(0,1,k*N+2); 
    x_global = xg(2:end-1);
    sigma_global = kSigma*(1/(k*N)) * ones(size(x_global));

    x_part = x_part(:);
    sigma_part = sigma_part(:);
    x_global = x_global(:);
    sigma_global = sigma_global(:);
end
