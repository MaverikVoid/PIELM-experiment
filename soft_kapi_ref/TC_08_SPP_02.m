clc; clear; close all;

%% ---------------------------------------------------------------
%   Test Case 8: Singularly Perturbed Convection–Diffusion
%   eps*u_xx + u_x + 1 = 0,   u(0)=0, u(1)=0
% ---------------------------------------------------------------

tic;  % ---------------- Timing start ----------------------------

%% Problem setup
epsPDE = 1e-4;
xL = 0; xR = 1;

%% Centers + widths from partition + global
N   = 1000;                   % points per partition (1D)
w   = 0.05;                   % tunable via BO
lens = [w, 1-w];              % positive, sums to 1
kSigma = 5.0;

[x_part, sigma_part, x_global, sigma_global, edges] = ...
    SAMP_POINTS_SIGMA(N, lens, kSigma);

% Combine centers and widths
alpha_star = [x_part; x_global];      % NN x 1
sig_x      = [sigma_part; sigma_global];
NN = numel(alpha_star);

%% Use centers as PDE collocation points
X_pde = sort(alpha_star);  
Nc    = NN;

%% Gaussian basis: φ(x) = exp(-(m x + α)^2)
m     = 1 ./ (sqrt(2) * sig_x);
alpha = -m .* alpha_star;

%% Build PDE residual: eps*φ_xx + φ_x + 1 = 0
z   = X_pde * m.' + alpha.';      
z2  = min(z.^2, 700);
phi = exp(-z2);

phi_x  = -2 .* (ones(Nc,1)*m.')    .* z   .* phi;
phi_xx =  2 .* (ones(Nc,1)*(m.'.^2)).*(2*z2 - 1).* phi;

LHS_PDE = epsPDE .* phi_xx + phi_x;
RHS_PDE = -ones(Nc,1);              % +1 moved to RHS

%% Boundary conditions
X_bc   = [xL; xR];
z_bc   = X_bc * m.' + alpha.';
phi_bc = exp(-min(z_bc.^2,700));

LHS_BC = phi_bc;
RHS_BC = [0; 0];

%% Assemble & solve
H = [LHS_PDE; LHS_BC];
b = [RHS_PDE; RHS_BC];

c = H \ b;      % QR-based least squares

J = norm(H*c - b, Inf);

%% Exact solution and PIELM prediction
u_exact = EXACT_SOLN_CD(X_pde, epsPDE);
u_pielm = phi * c;

solveTime = toc;     % ---------------- Timing end -----------------

fprintf('NN = %d, Nc = %d,  ||Hc-b||_inf = %.3e\n', NN, Nc, J);
fprintf('Solve time: %.4f seconds\n', solveTime);

%% ---------------------------------------------------------------
%   Figure settings
% ---------------------------------------------------------------
set(groot,'defaultTextInterpreter','latex');
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
set(groot,'defaultAxesFontSize',20);
set(groot,'defaultTextFontSize',24);

%% ---------------------------------------------------------------
%   Figure 1: Solution
% ---------------------------------------------------------------
fig1 = figure('Color','w','Position',[100 100 1400 500]);
plot(X_pde, u_pielm,'-r','LineWidth',2.4); hold on;
plot(X_pde, u_exact,'--b','LineWidth',2.4);
grid on; xlim([0 1]);
xlabel('$x$'); ylabel('$u$');
title(sprintf('KAPI-ELM vs exact solution, $\\epsilon = %.3g$', epsPDE));
legend({'KAPI-ELM','Exact'},'Location','best');

exportgraphics(fig1,'TC_08_Singular_Perturbation_Solution.png','Resolution',300);

%% ---------------------------------------------------------------
%   Figure 2: Centers
% ---------------------------------------------------------------
fig2 = figure('Color','w','Position',[100 100 1400 450]);
hold on;
scatter(x_part,   1+0*x_part,   16, 'b','filled','DisplayName','Partition centers');
scatter(x_global, 0+0*x_global, 16, 'r','filled','DisplayName','Global centers');
arrayfun(@(e) xline(e,'--k','HandleVisibility','off'), edges);
xlim([0 1]); ylim([-0.5 1.5]); grid on;
xlabel('$x$'); ylabel('row');
legend('Location','best');
title('Centers (partition vs global)');

% exportgraphics(fig2,'TC_08_Singular_Perturbation_Centers.png','Resolution',300);

%% ---------------------------------------------------------------
%   Figure 3: Sigmas
% ---------------------------------------------------------------
fig3 = figure('Color','w','Position',[100 100 1400 600]);

subplot(2,1,1);
plot(x_part, sigma_part, '.-','LineWidth',1.6); grid on;
xlabel('$x$'); ylabel('$\sigma$');
title('$\sigma$ (partition, piecewise-constant)');

subplot(2,1,2);
plot(x_global, sigma_global, '.-','LineWidth',1.6); grid on;
xlabel('$x$'); ylabel('$\sigma$');
title('$\sigma$ (global, constant)');

exportgraphics(fig3,'TC_08_Singular_Perturbation_Sigmas.png','Resolution',300);

%% ===============================================================
%   Helpers
% ===============================================================

function [x_part, sigma_part, x_global, sigma_global, edges] = SAMP_POINTS_SIGMA(N, customLens, kSigma)
    if nargin < 3 || isempty(kSigma), kSigma = 1.0; end
    customLens = customLens(:).';  
    k = numel(customLens);

    s = sum(customLens);  
    if abs(s-1) > 1e-12, customLens = customLens / s; end

    edges = [0, cumsum(customLens)]; edges(end)=1;

    % Partition centers
    x_part = [];
    for j = 1:k
        a = edges(j); b = edges(j+1);
        if j==1 && j==k
            pts = linspace(a,b,N);
        elseif j==1
            pts = linspace(a,b,N+1); pts = pts(1:N);
        elseif j==k
            pts = linspace(a,b,N+1); pts = pts(2:end);
        else
            pts = linspace(a,b,N+2); pts = pts(2:end-1);
        end
        x_part = [x_part, pts];
    end

    % Sigmas for partition
    sigma_part = zeros(1, k*N);
    idx = 0;
    for j = 1:k
        sigma_part(idx+(1:N)) = kSigma * (customLens(j)/N);
        idx = idx + N;
    end

    % Global centers and sigmas
    xg = linspace(0,1,k*N+2);  
    x_global = xg(2:end-1);
    sigma_global = kSigma * (1/(k*N)) * ones(size(x_global));

    x_part = x_part(:);
    sigma_part = sigma_part(:);
    x_global = x_global(:);
    sigma_global = sigma_global(:);
end

function u = EXACT_SOLN_CD(x, epsPDE)
    x = double(x);
    a = 1./double(epsPDE);

    denom = -expm1(-a);
    if ~isfinite(denom) || denom < 1e-300
        denom = 1.0;
    end

    t  = min(a .* x, 700);
    ex = exp(-t);

    u = (1./denom) - x - ex./denom;
end
