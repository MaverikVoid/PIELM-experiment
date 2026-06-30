clc; clear; close all;
rng(42);

%% ---------------- Manufactured solution setup ----------------
kx = 6*pi;                          % oscillation in x
ky = 6*pi;                          % oscillation in y
xL=0; xR=1; yB=0; yT=1;

u_exact_fun = @(x,y) sin(kx*x).*cos(ky*y);
f_handle    = @(x,y) -(kx^2+ky^2).*u_exact_fun(x,y);   % f = -(kx^2+ky^2) * u_exact

%% ---------------- Collocation via partitions + global ----------------
Nx = 20; Ny = 20;                   % points per partition (axes)
lensX = [0.5,0.5];           % partitions along x (sum=1)
lensY = [0.5,0.5];           % partitions along y (sum=1)
kSigma = 5.0;                       % sigma scale (σ = kSigma * sqrt(dx*dy))


[XY_part, sigma_part, XY_global, sigma_global, edgesX, edgesY] = ...
    SAMP_POINTS_SIGMA_2D(Nx, Ny, lensX, lensY, kSigma);

% RBF centers: use BOTH partition & global sets
XYc   = [XY_part; XY_global];                     % (NN x 2)
sigma = [sigma_part(:); sigma_global(:)];         % (NN x 1)
NN    = size(XYc,1);

% PDE collocation: use centers
X_pde = XYc;                                      % (Nc x 2)
Nc    = size(X_pde,1);

% Boundary collocation (Dirichlet values from u_exact) - DENSE
Nb_side = 8 * max(Nx, Ny);                        % more boundary points
xLine   = linspace(xL, xR, Nb_side).';
yLine   = linspace(yB, yT, Nb_side).';
X_lft   = [xL*ones(Nb_side,1), yLine];
X_ryt   = [xR*ones(Nb_side,1), yLine];
X_bot   = [xLine, yB*ones(Nb_side,1)];
X_top   = [xLine, yT*ones(Nb_side,1)];
X_bc    = [X_lft; X_ryt; X_bot; X_top];
Nb      = size(X_bc,1);

%% ---------------- RBF parameters (isotropic Gaussians) ----------------
% φ_i(x,y) = exp(-( (m_i x + α_i)^2 + (n_i y + β_i)^2 ))
% m_i = n_i = 1/(sqrt(2)*σ_i),  α_i = -m_i * x_i,  β_i = -n_i * y_i
m = 1 ./ (sqrt(2) * sigma);            % (NN x 1)
n = m;                                  % isotropic
alpha = -m .* XYc(:,1);                 % (NN x 1)
beta  = -n .* XYc(:,2);                 % (NN x 1)

%% ---------------- Build PDE rows (Laplacian) ----------------
% Δφ = -2 φ [ m^2 (1 - 2 z_x^2) + n^2 (1 - 2 z_y^2) ]
X = X_pde(:,1); Y = X_pde(:,2);
Zx = X.*m.' + alpha.';                          % Nc x NN
Zy = Y.*n.' + beta.';                           % Nc x NN
Z2 = Zx.^2 + Zy.^2;
Phi = exp(-min(Z2, 700));                       % clamp exponent

m2 = (m.'.^2); n2 = (n.'.^2);
LHS_PDE = -2 .* Phi .* ( (ones(Nc,1)*m2) .* (1 - 2*Zx.^2) + ...
                         (ones(Nc,1)*n2) .* (1 - 2*Zy.^2) );
RHS_PDE = f_handle(X, Y);                       % Nc x 1

%% ---------------- SDF-based PDE row weights ----------------
% Signed distance inside (positive): d = min{x,1-x,y,1-y}
d_pde = sdf_unit_square(X, Y);                  % Nc x 1

% Tunables: near/far weights, boundary layer width, smoothness
wPDE_near = 0.05;                                % PDE weight at boundary (d=0)
wPDE_far  = 1.00;                                % PDE weight in interior (d >= delta)
delta     = 0.08;                                % boundary layer width
p_smooth  = 2.0;                                 % sharpness of ramp

% Smooth ramp 0->1 across [0,delta] (use smoothstep_ease or power ramp)
t  = max(0, min(1, d_pde./delta));               % clamp to [0,1]
% Option A: power ramp
ramp = t.^p_smooth;
% Option B (swap in if you prefer): ramp = smoothstep_ease(t);  % C1 smoothstep

wPDE_vec = wPDE_near + (wPDE_far - wPDE_near).*ramp;  % Nc x 1

% Apply row-wise weights to PDE block
LHS_PDE = LHS_PDE .* (wPDE_vec .* ones(1,NN));
RHS_PDE = RHS_PDE .* wPDE_vec;

%% ---------------- Boundary rows: Dirichlet from u_exact ---------------
Xb = X_bc(:,1); Yb = X_bc(:,2);
Zx_b = Xb.*m.' + alpha.'; 
Zy_b = Yb.*n.' + beta.'; 
Phib = exp(-min(Zx_b.^2 + Zy_b.^2, 700));
LHS_BC = Phib;                                   % Nb x NN
RHS_BC = u_exact_fun(Xb, Yb);                    % Nb x 1

% Constant BC weight (still tunable)
% wBC = 40.0;
wPDE = 1.0; 
% wBC  = 50.0;
wBC = 1.0;

LHS_BC = wBC * LHS_BC;
RHS_BC = wBC * RHS_BC;

%% ---------------- Assemble and solve -------------------------
H = [LHS_PDE; LHS_BC];                           % ((Nc+Nb) x NN)
b = [RHS_PDE; RHS_BC];

% Least-squares solve (QR)
c = H \ b;
% Alternative: c = lsqminnorm(H, b);

resInf = norm(H*c - b, Inf);
fprintf('NN=%d, Nc=%d, Nb=%d,  ||Hc-b||_inf = %.3e\n', NN, Nc, Nb, resInf);

%% ---------------- Evaluate on plotting grid and compare ---------------
Nplot = 160;
xg = linspace(xL, xR, Nplot);
yg = linspace(yB, yT, Nplot);
[XX, YY] = meshgrid(xg, yg);

Zx_t = XX(:).*m.' + alpha.'; 
Zy_t = YY(:).*n.' + beta.'; 
Ut   = exp(-min(Zx_t.^2 + Zy_t.^2, 700)) * c;   % predicted u (vector)
U_pred = reshape(Ut, Nplot, Nplot);

U_exact = u_exact_fun(XX, YY);

abs_err = abs(U_pred - U_exact);
mse = mean(abs_err.^2, 'all');
fprintf('MSE on %dx%d grid: %.3e\n', Nplot, Nplot, mse);

%% ---------------- Plots ----------------
figure('Color','w','Position',[100 100 1800 650]);

subplot(2,3,1);
scatter(X_pde(:,1), X_pde(:,2), 9, wPDE_vec, 'filled'); axis equal tight;
title('PDE row weight $w_{\mathrm{PDE}}(x,y)$','Interpreter','latex'); colorbar;
xlabel('x'); ylabel('y'); grid on;

subplot(2,3,2);
plot( linspace(0,delta,200), ...
      wPDE_near + (wPDE_far-wPDE_near)*(linspace(0,1,200).^p_smooth), 'LineWidth',2);
xline(delta,'--k'); ylim([0,1.05]); grid on;
% title(sprintf('Ramp vs distance (\\delta=%.2f, p=%.1f)',delta,p_smooth));
title(['Ramp vs distance ($\delta = ', num2str(delta,'%.2f'), ...
',\ p = ', num2str(p_smooth,'%.1f'), '$)'], ...
'Interpreter','latex');
xlabel('distance to boundary d'); ylabel('$w_{PDE}$');

subplot(2,3,3);
scatter(Xb, Yb, 6, 'k', 'filled'); axis equal tight; grid on;
title(sprintf('Boundary points (Nb\\_side = %d)', Nb_side));
xlabel('x'); ylabel('y');

subplot(2,3,4);
surf(XX, YY, U_pred,'EdgeColor','none'); view(2); grid on; colorbar;
title('KAPI-ELM solution'); xlabel('x'); ylabel('y');

subplot(2,3,5);
surf(XX, YY, U_exact,'EdgeColor','none'); view(2); grid on; colorbar;
title('Manufactured exact'); xlabel('x'); ylabel('y');

subplot(2,3,6);
surf(XX, YY, abs_err,'EdgeColor','none'); view(2); grid on; colorbar;
title(sprintf('Absolute error (MSE=%.2e)', mse)); xlabel('x'); ylabel('y');
exportgraphics(gcf,'TC_04_Comparison.png','Resolution',300);
%--------------------------------------------------------------------------------------------------
% Create a new figure for the error subplot only
fig_error = figure('Color','w','Position',[100 100 800 650]); % Adjust size/aspect ratio as desired

% Plot the absolute error surface again
surf(XX, YY, abs_err, 'EdgeColor', 'none');
view(2);
grid on;
colorbar;
title(sprintf('Absolute error (MSE=%.2e)', mse));
xlabel('x');
ylabel('y');

% Save this figure as error.png with high resolution
exportgraphics(fig_error, 'TC_04_Comparison_error.png', 'Resolution', 300);

% (Optional) Close the temporary figure after saving
close(fig_error);
%% ===================== Sampler (partition + global) ===================
function [XY_part, sigma_part, XY_global, sigma_global, edgesX, edgesY] = ...
    SAMP_POINTS_SIGMA_2D(Nx, Ny, lensX, lensY, kSigma)
% Returns centers & isotropic σ for partition- and global-grids on [0,1]^2.
% σ_part(i,j) = kSigma * sqrt((lensX_i/Nx)*(lensY_j/Ny)),  σ_global = const.

    if nargin < 5 || isempty(kSigma), kSigma = 1.0; end

    lensX = lensX(:).'; lensY = lensY(:).';
    if isempty(lensX) || any(lensX <= 0), error('lensX must be positive and non-empty.'); end
    if isempty(lensY) || any(lensY <= 0), error('lensY must be positive and non-empty.'); end
    sx = sum(lensX); sy = sum(lensY);
    if abs(sx - 1) > 1e-12, lensX = lensX / sx; end
    if abs(sy - 1) > 1e-12, lensY = lensY / sy; end
    kx = numel(lensX); ky = numel(lensY);

    % Edges
    edgesX = [0, cumsum(lensX)]; edgesX(end) = 1;
    edgesY = [0, cumsum(lensY)]; edgesY(end) = 1;

    % 1D axes (dedup at interior edges) + partition indices
    [x_axis, x_part_idx] = build_axis_with_idx(edgesX, Nx); % length kx*Nx
    [y_axis, y_part_idx] = build_axis_with_idx(edgesY, Ny); % length ky*Ny

    % Partition grid
    [Xp, Yp] = meshgrid(x_axis, y_axis);
    XY_part = [Xp(:), Yp(:)];

    % Partition indices per grid point
    [IX, IY] = meshgrid(x_part_idx, y_part_idx);
    IX = IX(:); IY = IY(:);

    % Partition σ (isotropic via geometric mean of local spacings)
    dx_local = lensX(IX) / Nx;
    dy_local = lensY(IY) / Ny;
    sigma_part = kSigma * sqrt(dx_local .* dy_local);
    sigma_part = sigma_part(:);                         % column-safe

    % Global grid (exclude endpoints to avoid duplicates)
    xg = linspace(0, 1, kx*Nx + 2); xg = xg(2:end-1);
    yg = linspace(0, 1, ky*Ny + 2); yg = yg(2:end-1);
    [Xg, Yg] = meshgrid(xg, yg);
    XY_global = [Xg(:), Yg(:)];

    % Global σ (constant)
    dxg = 1 / (kx * Nx);
    dyg = 1 / (ky * Ny);
    sigma_global = kSigma * sqrt(dxg * dyg) * ones(size(XY_global,1), 1);
    sigma_global = sigma_global(:);                    % column-safe
end

function [axis_vals, part_idx] = build_axis_with_idx(edges, N)
    k = numel(edges) - 1;
    axis_vals = [];
    part_idx  = [];
    for i = 1:k
        a = edges(i); b = edges(i+1);
        if k == 1
            v = linspace(a, b, N);
        elseif i == 1
            v = linspace(a, b, N+1); v = v(1:N);       % include left, exclude right
        elseif i == k
            v = linspace(a, b, N+1); v = v(2:end);     % exclude left, include right
        else
            v = linspace(a, b, N+2); v = v(2:end-1);   % exclude both
        end
        axis_vals = [axis_vals, v]; %#ok<AGROW>
        part_idx  = [part_idx, i * ones(1, numel(v))]; %#ok<AGROW>
    end
end

%% ==================== SDF + smoothstep helpers =======================
function d = sdf_unit_square(x, y)
% Inside-distance to boundary for (0,1)^2 (non-negative, zero on ∂Ω)
    d = min( min(x, 1-x), min(y, 1-y) );
end

function s = smoothstep_ease(t)
% C1-smooth step: 3t^2 - 2t^3, for t in [0,1]
    t = max(0, min(1, t));
    s = t.^2 .* (3 - 2*t);
end
