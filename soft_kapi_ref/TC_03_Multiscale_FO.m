clc; clear; close all;

%% ---------------- Problem: Multiscale First-Order ODE ----------------
w1 = 1;                           % low frequency component
w2 = 15;                          % high frequency component
xL = -2*pi; xR =  2*pi;           % domain [-2π, 2π]
bc_x  = 0; 
bc_val = 0;                       % u(0) = 0

%% --------- Centers + widths (partition + global) ----------
N       = 400;                    % points per partition
w       = 0.50;                   % partition weight
lens    = [w, 1-w];
kSigma  = 2.0;

[x_part, sig_part, x_glob, sig_glob, edges] = ...
    SAMP_POINTS_SIGMA_DOMAIN(N, lens, kSigma, xL, xR);

alpha_star = [x_part; x_glob];
sig_x      = [sig_part; sig_glob];
NN         = numel(alpha_star);

X_pde = sort(alpha_star);   
Nc    = NN;

%% ---------------- Gaussian basis -----------------
m     = 1 ./ (sqrt(2) * sig_x);
alpha = -m .* alpha_star;

z   = X_pde * m.' + alpha.';    
z2  = min(z.^2, 700);
phi = exp(-z2);

phi_x = -2 .* (ones(Nc,1)*m.') .* z .* phi;

%% ----------------- PDE & BC rows -----------------
LHS_PDE = phi_x;
RHS_PDE = w1*cos(w1*X_pde) + w2*cos(w2*X_pde);

z_bc   = bc_x * m.' + alpha.';
phi_bc = exp(-min(z_bc.^2,700));

LHS_BC = phi_bc;         
RHS_BC = bc_val;

wPDE = 1.0; 
wBC  = 50.0;

H = [wPDE*LHS_PDE; wBC*LHS_BC];
b = [wPDE*RHS_PDE; wBC*RHS_BC];

%% ---------------- Solve least squares ----------------
tic;
c = H \ b;                       % QR-based LS
t_solve = toc;
fprintf('Solve time: %.4f seconds\n', t_solve);

J = norm(H*c - b, Inf);
fprintf('NN=%d, Nc=%d,  ||Hc-b||_inf = %.3e\n', NN, Nc, J);

%% ---------------- Evaluate & Publication-ready Plot -------------------
u_pred  = phi * c;
u_exact = sin(w1*X_pde) + sin(w2*X_pde);

fig = figure('Color','w','Units','inches','Position',[1 1 10 3]);
hold on;
plot(X_pde, u_pred,'-r','LineWidth',1.8);
plot(X_pde, u_exact,'--b','LineWidth',1.5);

grid on; xlim([xL xR]);
xlabel('$x$','Interpreter','latex','FontSize',18);
ylabel('$u(x)$','Interpreter','latex','FontSize',18);

title(sprintf('KAPI--ELM approximation for $u''(x)=\\omega_1\\cos(\\omega_1 x)+\\omega_2\\cos(\\omega_2 x)$, $[\\omega_1,\\omega_2]=[%g,%g]$', w1, w2), ...
    'Interpreter','latex','FontSize',18);

legend({'KAPI--ELM','Exact'},'Interpreter','latex','FontSize',16,'Location','best');
set(gca,'TickLabelInterpreter','latex','FontSize',14);

saveas(fig,'TC_03_Comparison.png');

%% ================= Helper Function =================
function [x_part, sigma_part, x_global, sigma_global, edges_x] = ...
    SAMP_POINTS_SIGMA_DOMAIN(N, customLens, kSigma, xL, xR)

    if nargin < 3 || isempty(kSigma), kSigma = 1.0; end
    customLens = customLens(:).'; 
    if any(customLens <= 0), error('Partition lengths must be > 0.'); end
    s = sum(customLens); 
    if abs(s-1) > 1e-12, customLens = customLens / s; end
    L = (xR - xL);

    edges = [0, cumsum(customLens)]; edges(end)=1;
    edges_x = xL + L * edges;

    xi_part = [];
    k = numel(customLens);
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
        xi_part = [xi_part, pts];
    end
    x_part = xL + L * xi_part(:);

    sigma_part = zeros(k*N,1);
    idx = 0;
    for j = 1:k
        sigma_part(idx+(1:N)) = kSigma * ((customLens(j)*L)/N);
        idx = idx + N;
    end

    xg = linspace(0,1,k*N+2);
    xi_global = xg(2:end-1).';
    x_global  = xL + L * xi_global;
    sigma_global = kSigma * (L/(k*N)) * ones(k*N,1);
end
