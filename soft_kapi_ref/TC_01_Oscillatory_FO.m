clc; clear; close all;

%% ---------------- Problem setup ----------------
omega = 15;                         % try 1 or 15 (like the paper)
xL = -2*pi; xR = 2*pi;              % domain [-2π, 2π]
bc_x = 0; bc_val = 0;               % u(0) = 0

%% ------------- Centers + widths (partition + global) -------------
% Tunables (can be optimized later)
N       = 400;                      % points per partition (per 1D piece)
w       = 0.50;                     % partition weight along [0,1] (e.g., 0.5 -> two equal halves)
lens    = [w, 1-w];                 % positive, sums to 1
kSigma  = 2.0;                      % scale in sigma = kSigma * (length/points)

% Centers/widths on [xL, xR]
[x_part, sigma_part, x_global, sigma_global, edges] = ...
    SAMP_POINTS_SIGMA_DOMAIN(N, lens, kSigma, xL, xR);

% Combine (partition first, then global)
alpha_star = [x_part; x_global];                % NN x 1 centers
sig_x      = [sigma_part; sigma_global];        % NN x 1 widths
NN         = numel(alpha_star);

% Use centers as PDE collocation points (can also add extra points if desired)
X_pde = sort(alpha_star);  Nc = NN;

%% ----------------- Gaussian basis parameters ---------------------
% phi_i(x) = exp(-(m_i x + alpha_i)^2),     m_i = 1/(sqrt(2)*sigma_i), alpha_i = -m_i * center_i
m     = 1 ./ (sqrt(2) * sig_x);            % NN x 1
alpha = -m .* alpha_star;                  % NN x 1

% Build matrices at PDE points
z   = X_pde * m.' + alpha.';               % (Nc x NN)
z2  = min(z.^2, 700);                      % clamp exponent for safety
phi = exp(-z2);                            

% phi_x = -2*m*z .* phi
phi_x = -2 .* (ones(Nc,1)*m.') .* z .* phi;

% ODE residual:  phi_x * c = cos(omega * x)
LHS_PDE = phi_x;                           % (Nc x NN)
RHS_PDE = cos(omega * X_pde);              % (Nc x 1)

%% ---------------------- Boundary condition -----------------------
% u(bc_x) = bc_val
z_bc   = bc_x * m.' + alpha.';             % (1 x NN)
phi_bc = exp(-min(z_bc.^2,700));
LHS_BC = phi_bc;                           % (1 x NN)
RHS_BC = bc_val;                           % scalar

%% ------------------------ Assemble & solve -----------------------
wPDE = 1.0; 
wBC  = 50.0;

H = [wPDE*LHS_PDE; wBC*LHS_BC];
b = [wPDE*RHS_PDE; wBC*RHS_BC];

tic;
c = H \ b;               % QR-based least-squares solve
t_solve = toc;
fprintf('Solve time: %.4f seconds\n', t_solve);

J = norm(H*c - b, Inf);
fprintf('NN=%d, Nc=%d,  ||Hc-b||_inf = %.3e\n', NN, Nc, J);

%% ------------------------ Evaluate & plot ------------------------
u_exact = (1/omega) * sin(omega * X_pde);
u_pred  = phi * c;

fig = figure('Color','w','Units','inches','Position',[1 1 10 3]);  % wide figure
hold on;
plot(X_pde, u_pred, '-r', 'LineWidth', 1.8);
plot(X_pde, u_exact,'--b','LineWidth', 1.5);

grid on; xlim([xL xR]);

xlabel('$x$','Interpreter','latex','FontSize',18);
ylabel('$u(x)$','Interpreter','latex','FontSize',18);
title(sprintf('KAPI-ELM approximation for $u''(x)=\\cos(\\omega x)$, $\\omega=%g$',omega), ...
    'Interpreter','latex','FontSize',18);
legend({'KAPI-ELM','Exact'},'Interpreter','latex','FontSize',16,'Location','best');

set(gca,'TickLabelInterpreter','latex','FontSize',14);

% save figure
saveas(fig,'TC_01_Comparison.png');


% Visualize centers and sigmas (on the actual domain)
figure('Color','w'); hold on;
scatter(x_part,   1+0*x_part,   12, 'b', 'filled', 'DisplayName','Partition centers');
scatter(x_global, 0+0*x_global, 12, 'r', 'filled', 'DisplayName','Global centers');
arrayfun(@(e) xline(e,'--k','HandleVisibility','off'), edges);
xlim([xL xR]); ylim([-0.5 1.5]); grid on;
xlabel('x'); ylabel('row'); legend('Location','best');
title('Centers (partition vs global)');

figure('Color','w');
subplot(2,1,1); plot(x_part,   sigma_part,   '.-'); grid on; xlim([xL xR]);
xlabel('x'); ylabel('\sigma'); title('\sigma (partition, piecewise-constant)');
subplot(2,1,2); plot(x_global, sigma_global, '.-'); grid on; xlim([xL xR]);
xlabel('x'); ylabel('\sigma'); title('\sigma (global, constant)');

%% ===================== Helper functions ==========================
function [x_part, sigma_part, x_global, sigma_global, edges_x] = ...
    SAMP_POINTS_SIGMA_DOMAIN(N, customLens, kSigma, xL, xR)
% Centers + widths on a general 1D domain [xL, xR].
% Internally samples on ξ∈[0,1] using customLens, then maps x = xL + (xR-xL)*ξ.
% Sigma scales with domain length s = (xR-xL).

    if nargin < 3 || isempty(kSigma), kSigma = 1.0; end
    if nargin < 5, error('Provide N, customLens, kSigma, xL, xR.'); end
    customLens = customLens(:).';  k = numel(customLens);
    if any(customLens <= 0), error('All partition lengths must be > 0.'); end
    s = sum(customLens); if abs(s-1) > 1e-12, customLens = customLens / s; end
    s_dom = (xR - xL);                      % domain length

    % Edges in ξ and mapped edges in x
    edges = [0, cumsum(customLens)]; edges(end) = 1;
    edges_x = xL + s_dom * edges;

    % Partition centers in ξ (no dupes at interior edges)
    xi_part = [];
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
        xi_part = [xi_part, pts]; %#ok<AGROW>
    end

    % Map to x-domain
    x_part = xL + s_dom * xi_part(:);

    % Partition sigmas (in x-units):  kSigma * (len_x / N)
    sigma_part = zeros(k*N,1);
    idx = 0;
    for j = 1:k
        len_x_j = customLens(j) * s_dom;
        sigma_part(idx + (1:N)) = kSigma * (len_x_j / N);
        idx = idx + N;
    end

    % Global grid: k*N points in (0,1) excluding endpoints, then map to [xL,xR]
    xg = linspace(0, 1, k*N + 2);
    xi_global = xg(2:end-1).';
    x_global  = xL + s_dom * xi_global;

    % Global sigma (in x-units): kSigma * (s_dom / (k*N))
    sigma_global = kSigma * (s_dom / (k*N)) * ones(k*N,1);
end
