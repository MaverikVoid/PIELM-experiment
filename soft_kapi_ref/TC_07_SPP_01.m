clc; clear; close all;
rng(42);

%% ---------------- Timing start ----------------
tic;

%% Problem setup
nu  = 1e-4;
xL  = 0;  xR  = 1;

%% Centers + widths from partition + global (your rule)
N   = 1000;              % points per partition (1D)
w   = 0.95;              % CAN BE TUNED WITH BAYESIAN OPTIMIZATION
lens = [w, 1-w];         % positive, sums to 1
kSigma = 5.0;

[x_part, sigma_part, x_global, sigma_global, edges] = ...
    SAMP_POINTS_SIGMA(N, lens, kSigma);

% Combine centers and widths (partition first, then global)
alpha_star = [x_part;       x_global      ];   % NN x 1
sig_x      = [sigma_part;   sigma_global  ];   % NN x 1
NN         = numel(alpha_star);

%% Use alpha_star as PDE collocation points
X_pde = sort(alpha_star);                        % Nc x 1
Nc    = NN;

%% ELM parameters for Gaussian basis: phi(x) = exp(-(m x + alpha)^2)
m     = 1 ./ (sqrt(2) * sig_x);           % (NN x 1)
alpha = -m .* alpha_star;                 % (NN x 1)

%% Build PDE residual matrix H (u_x - nu*u_xx = 0)
z      = X_pde * m.' + alpha.';           % (Nc x NN)
z2     = min(z.^2, 700);
phi    = exp(-z2);                         % (Nc x NN)

% Derivatives
phi_x  = -2 .* (ones(Nc,1)*m.') .* z .* phi;
phi_xx =  2 .* (ones(Nc,1)*(m.'.^2)) .* (2*z2 - 1) .* phi;

% PDE operator: L{phi} = phi_x - nu*phi_xx
LHS_PDE = phi_x - nu .* phi_xx;
RHS_PDE = zeros(Nc,1);

%% Boundary conditions: u(0)=0, u(1)=1
X_bc   = [xL; xR];
z_bc   = X_bc * m.' + alpha.';
phi_bc = exp(-z_bc.^2);
LHS_BC = phi_bc;
RHS_BC = [0; 1];

%% Assemble and solve (OLS)
H = [LHS_PDE; LHS_BC];
b = [RHS_PDE; RHS_BC];

c = H \ b;

%% Residual metric
J = norm(H*c - b, Inf);
fprintf('Total centers (NN): %d | Nc (PDE pts) = %d | ||Hc-b||_inf = %.3e\n', NN, Nc, J);
%% ---------------- Timing end ----------------
solve_time = toc;
fprintf('Solve time: %.4f seconds\n', solve_time);

%% Compare against exact solution
u_exact = EXACT_SOLUTION(X_pde, nu);
u_pielm = phi * c;

%% ---------------- Plot settings (LaTeX + big fonts) ----------------
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
set(groot,'defaultTextInterpreter','latex');
set(groot,'defaultAxesFontSize',22);
set(groot,'defaultTextFontSize',24);

%% ---------------- Figure 1: Solution ----------------
fig1 = figure('Color','w','Position',[100 100 1400 500]);
plot(X_pde, u_pielm,'-r','LineWidth',2.4); hold on;
plot(X_pde, u_exact,'--b','LineWidth',2.4);
grid on; xlim([0 1]);
xlabel('$x$'); ylabel('$u$');
title(sprintf('KAPI-ELM vs exact solution, $\\nu = %.3g$', nu));
legend({'KAPI-ELM','Exact'},'Location','best');
exportgraphics(fig1,'TC_07_Singular_Perturbation_Solution.png','Resolution',300);

%% ---------------- Figure 2: Centers ----------------
fig2 = figure('Color','w','Position',[100 100 1400 450]);
hold on;
scatter(x_part,   1.0*ones(size(x_part)),   16, 'b', 'filled', 'DisplayName','Partition centers');
scatter(x_global, 0.0*ones(size(x_global)), 16, 'r', 'filled', 'DisplayName','Global centers');
arrayfun(@(e) xline(e,'--k','HandleVisibility','off'), edges);
xlim([0 1]); ylim([-0.5 1.5]); grid on;
xlabel('$x$'); ylabel('row');
legend('Location','best');
title('Centers (partition vs global)');
% exportgraphics(fig2,'TC_07_Singular_Perturbation_Centers.png','Resolution',300);

%% ---------------- Figure 3: Sigmas ----------------
fig3 = figure('Color','w','Position',[100 100 1400 600]);

subplot(2,1,1);
plot(x_part, sigma_part, '.-', 'LineWidth', 1.6); grid on;
xlabel('$x$', 'Interpreter', 'latex');
ylabel('$\sigma$', 'Interpreter', 'latex');
title('$\sigma$ (partition, piecewise-constant)', 'Interpreter', 'latex');

subplot(2,1,2);
plot(x_global, sigma_global, '.-', 'LineWidth', 1.6); grid on;
xlabel('$x$', 'Interpreter', 'latex');
ylabel('$\sigma$', 'Interpreter', 'latex');
title('$\sigma$ (global, constant)', 'Interpreter', 'latex');

exportgraphics(fig3, 'TC_07_Singular_Perturbation_Sigmas.png', 'Resolution', 300);

%% ============================================================
%% Helpers
%% ============================================================
function [x_part, sigma_part, x_global, sigma_global, edges] = SAMP_POINTS_SIGMA(N, customLens, kSigma)
    if nargin < 3 || isempty(kSigma), kSigma = 1.0; end
    customLens = customLens(:).';  k = numel(customLens);
    s = sum(customLens); if abs(s-1) > 1e-12, customLens = customLens / s; end
    edges = [0, cumsum(customLens)]; edges(end)=1;

    x_part = [];
    for j = 1:k
        a = edges(j); b = edges(j+1);
        if j == 1 && j == k
            pts = linspace(a, b, N);
        elseif j == 1
            pts = linspace(a, b, N+1); pts = pts(1:N);
        elseif j == k
            pts = linspace(a, b, N+1); pts = pts(2:end);
        else
            pts = linspace(a, b, N+2); pts = pts(2:end-1);
        end
        x_part = [x_part, pts];
    end

    sigma_part = zeros(1, k*N);
    idx=0;
    for j=1:k
        sig_j = kSigma*(customLens(j)/N);
        sigma_part(idx+(1:N)) = sig_j;
        idx = idx + N;
    end

    xg = linspace(0,1,k*N+2); x_global = xg(2:end-1);
    sigma_global = kSigma*(1/(k*N)) * ones(size(x_global));

    x_part = x_part(:);
    sigma_part = sigma_part(:);
    x_global = x_global(:);
    sigma_global = sigma_global(:);
end

function u_exact = EXACT_SOLUTION(X, nu)
    overflow_threshold = 1/log(realmax('double'));
    if nu > overflow_threshold
        u_exact = expm1(X./nu) ./ expm1(1/nu);
    else
        exponent = (X - 1)./nu;
        threshold = -log(eps(class(X)));
        u_exact = exp(exponent);
        u_exact(exponent < -threshold) = 0;
        u_exact(X==1) = 1;
    end
end
