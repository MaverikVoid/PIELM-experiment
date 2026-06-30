clc; clear; close all;

%% ------------ Problem ------------
omega = 15;                    % try 1 or 15
xL = -2*pi; xR =  2*pi;        % domain
bc_x  = 0;                     % BCs applied at x=0
bc_u  = 0;                     % Dirichlet u(0)=0
bc_du = -1/omega;              % Neumann  u'(0)=-1/omega

%% ----- Centers + widths (partition + global, in physical units) -----
N      = 400;                  % points per partition
w      = 0.50;                 % partition weight in [0,1]
lens   = [w, 1-w];
kSigma = 2.0;                  % sigma scale (tuneable)

[x_part, sig_part, x_glob, sig_glob, edges] = ...
    SAMP_POINTS_SIGMA_DOMAIN(N, lens, kSigma, xL, xR);

alpha_star = [x_part; x_glob];                 % NN x 1 centers
sig_x      = [sig_part; sig_glob];             % NN x 1 widths
NN         = numel(alpha_star);

% Collocation points for PDE: use centers
X_pde = sort(alpha_star); Nc = NN;

%% -------- Gaussian basis  φ_i(x)=exp(-(m_i x + α_i)^2) --------
m     = 1 ./ (sqrt(2) * sig_x);                % NN x 1
alpha = -m .* alpha_star;                      % NN x 1

% Build at PDE points
z   = X_pde * m.' + alpha.';                   % (Nc x NN)
z2  = min(z.^2, 700);                          % clamp for safety
phi = exp(-z2);

% φ_x = -2 m z φ
phi_x  = -2 .* (ones(Nc,1)*m.') .* z .* phi;

% φ_xx = 2 m^2 (2 z^2 - 1) φ
phi_xx =  2 .* (ones(Nc,1)*(m.'.^2)) .* (2*z2 - 1) .* phi;

%% ---------------------- PDE rows ----------------------
% u'' = sin(ωx)  →  φ_xx * c = sin(ω x)
LHS_PDE = phi_xx;                               % (Nc x NN)
RHS_PDE = sin(omega * X_pde);                   % (Nc x 1)

%% ---------------------- BC rows @ x=0 -----------------
z0    = bc_x * m.' + alpha.';                   % (1 x NN)
z0_2  = min(z0.^2,700);
phi0  = exp(-z0_2);                             % u(0) row
phi0_x = -2 .* m.' .* z0 .* phi0;               % u'(0) row  (1 x NN)

LHS_BC  = [phi0;            % u(0)=0
           phi0_x];         % u'(0) = -1/ω
RHS_BC  = [bc_u; bc_du];

%% ------------------ Assemble & solve ------------------
wPDE = 1.0;
wB1  = 50.0;      % Dirichlet BC weight
wB2  = 50.0;      % Neumann  BC weight

H = [wPDE*LHS_PDE;
     wB1 * LHS_BC(1,:);
     wB2 * LHS_BC(2,:)];

b = [wPDE*RHS_PDE;
     wB1 * RHS_BC(1);
     wB2 * RHS_BC(2)];

tic;
c = H \ b;                 % QR-based LS solve
t_solve = toc;
fprintf('Solve time: %.4f seconds\n', t_solve);

J = norm(H*c - b, Inf);
fprintf('NN=%d, Nc=%d, ||Hc-b||_inf = %.3e\n', NN, Nc, J);

%% ---------------- Evaluate & publication-ready plot ----------------
u_pred  = phi * c;
u_exact = -(1/omega^2) * sin(omega * X_pde);   % exact solution

fig = figure('Color','w','Units','inches','Position',[1 1 10 3]);
hold on;
plot(X_pde, u_pred,'-r','LineWidth',1.8);
plot(X_pde, u_exact,'--b','LineWidth',1.5);

grid on; xlim([xL xR]);
xlabel('$x$','Interpreter','latex','FontSize',18);
ylabel('$u(x)$','Interpreter','latex','FontSize',18);

title(sprintf('KAPI--ELM approximation for $u''''(x)=\\sin(\\omega x)$, $\\omega=%g$',omega), ...
    'Interpreter','latex','FontSize',18);


legend({'KAPI--ELM','Exact'},'Interpreter','latex','FontSize',16,'Location','best');
set(gca,'TickLabelInterpreter','latex','FontSize',14);

saveas(fig,'TC_02_Comparison.png');


% Centers & sigmas
figure('Color','w'); hold on;
scatter(x_part,   1+0*x_part,   12, 'b', 'filled', 'DisplayName','Partition centers');
scatter(x_glob,   0+0*x_glob,   12, 'r', 'filled', 'DisplayName','Global centers');
arrayfun(@(e) xline(e,'--k','HandleVisibility','off'), edges);
xlim([xL xR]); ylim([-0.5 1.5]); grid on; legend('Location','best');
xlabel('x'); ylabel('row'); title('Centers (partition vs global)');

figure('Color','w');
subplot(2,1,1); plot(x_part, sig_part, '.-'); grid on; xlim([xL xR]);
xlabel('x'); ylabel('\sigma'); title('\sigma (partition)');
subplot(2,1,2); plot(x_glob, sig_glob, '.-'); grid on; xlim([xL xR]);
xlabel('x'); ylabel('\sigma'); title('\sigma (global)');

%% ================= Helpers =================
function [x_part, sigma_part, x_global, sigma_global, edges_x] = ...
    SAMP_POINTS_SIGMA_DOMAIN(N, customLens, kSigma, xL, xR)
% Centers + widths on [xL,xR]; partitions defined on ξ∈[0,1], mapped by x=xL+(xR-xL)ξ.

    if nargin < 3 || isempty(kSigma), kSigma = 1.0; end
    customLens = customLens(:).'; k = numel(customLens);
    if any(customLens <= 0), error('All partition lengths must be > 0.'); end
    s = sum(customLens); if abs(s-1) > 1e-12, customLens = customLens / s; end
    L = (xR - xL);

    edges = [0, cumsum(customLens)]; edges(end)=1;
    edges_x = xL + L * edges;

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
