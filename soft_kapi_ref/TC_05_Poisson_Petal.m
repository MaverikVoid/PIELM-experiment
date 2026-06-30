clc; clear; close all;
rng(42);

%% ---------------- Timing start ----------------
tic;

%% ---------------- Manufactured solution ----------------
kx = 4*pi;
ky = 4*pi;
u_exact_fun = @(x,y) sin(kx*x).*cos(ky*y);
f_handle    = @(x,y) -(kx^2+ky^2).*u_exact_fun(x,y);

%% ---------------- Irregular domain via numerical SDF ----------------
Nsdf = 600;
[xg, yg, sdf, inMask] = build_irregular_domain_SDF(Nsdf);

sdfF = griddedInterpolant({xg, yg}, sdf.', 'linear',  'nearest');
mskF = griddedInterpolant({xg, yg}, double(inMask).','nearest','nearest');

xg = double(xg); yg = double(yg); sdf = double(sdf);
try
    B = contourc(xg, yg, sdf.', [0 0]);
    XYb_raw = contourc_to_points(B);
    if isempty(XYb_raw), error('Empty contour'); end
catch
    eps_band = max(1e-6, 2*mean(diff(xg)));
    BW = abs(sdf) <= eps_band;
    XYb_raw = bw_to_points(BW, xg, yg);
end

Nb_side = 2000;
XYb = resample_polyline_closed(XYb_raw, Nb_side);
X_bc = XYb;
Nb = size(X_bc,1);

%% ---------------- Centers: partition + global, then clip to Ω --------
Nx = 20; Ny = 20;
lensX = [0.5,0.5];
lensY = [0.5,0.5];
kSigma = 5.0;

[XY_part, sigma_part, XY_global, sigma_global] = ...
    SAMP_POINTS_SIGMA_2D(Nx, Ny, lensX, lensY, kSigma);

XYc_all = [XY_part; XY_global];
sig_all = [sigma_part(:); sigma_global(:)];
inside  = mskF(XYc_all(:,1), XYc_all(:,2)) > 0.5;
XYc     = XYc_all(inside,:);
sigma   = sig_all(inside);
NN      = size(XYc,1);

X_pde = XYc; 
Nc    = size(X_pde,1);

%% ---------------- RBF parameters (isotropic) ----------------
m = 1 ./ (sqrt(2)*sigma);
n = m;
alpha = -m .* XYc(:,1);
beta  = -n .* XYc(:,2);

%% ---------------- PDE rows with SDF weighting ----------------
X = X_pde(:,1); Y = X_pde(:,2);
Zx = X.*m.' + alpha.';
Zy = Y.*n.' + beta.';
Z2 = Zx.^2 + Zy.^2;

Phi = exp(-min(Z2,700));

m2 = m.'.^2; n2 = n.'.^2;
LHS_PDE = -2 .* Phi .* ( (ones(Nc,1)*m2).*(1 - 2*Zx.^2) + ...
                         (ones(Nc,1)*n2).*(1 - 2*Zy.^2) );
RHS_PDE = f_handle(X,Y);

%% --- SDF weights
d_pde = max(0, sdfF(X,Y));
wNear = 0.05; wFar = 1.0; delta = 0.1; p = 3.0;
t = max(0,min(1,d_pde./delta));
ramp = t.^p;

wPDE_vec = wNear + (wFar-wNear).*ramp;

LHS_PDE = LHS_PDE .* (wPDE_vec .* ones(1,NN));
RHS_PDE = RHS_PDE .* wPDE_vec;

%% ---------------- Boundary rows ----------------
Xb = X_bc(:,1); Yb = X_bc(:,2);
Zx_b = Xb.*m.' + alpha.';
Zy_b = Yb.*n.' + beta.';
Phib = exp(-min(Zx_b.^2 + Zy_b.^2, 700));

LHS_BC = Phib;
RHS_BC = u_exact_fun(Xb, Yb);

% wBC = 100.0;
wBC = 10.0;
LHS_BC = wBC * LHS_BC;
RHS_BC = wBC * RHS_BC;

%% ---------------- Assemble & solve ----------------
H = [LHS_PDE; LHS_BC];
b = [RHS_PDE; RHS_BC];

c = H \ b;

resInf = norm(H*c - b, Inf);
fprintf('NN=%d, Nc=%d, Nb=%d, ||Hc-b||_inf = %.3e\n', NN, Nc, Nb, resInf);

%% ---------------- Evaluate on fine grid ----------------
Nplot = 220;
xp = linspace(0,1,Nplot);
yp = linspace(0,1,Nplot);
[XX,YY] = meshgrid(xp,yp);
INS = mskF(XX,YY) > 0.5;

Zx_t = XX(:).*m.' + alpha.';
Zy_t = YY(:).*n.' + beta.';
Ut = exp(-min(Zx_t.^2 + Zy_t.^2,700))*c;

U_pred = reshape(Ut,Nplot,Nplot); U_pred(~INS) = NaN;
U_exact = u_exact_fun(XX,YY); U_exact(~INS) = NaN;

abs_err = abs(U_pred - U_exact);
mse = mean(abs_err(INS).^2,'all','omitnan');
fprintf('MSE on %dx%d masked grid: %.3e\n', Nplot,Nplot,mse);

%% ---------------- Timing end ----------------
solveTime = toc;
fprintf('Solve time: %.4f seconds\n', solveTime);

%% ---------------- Plots (LaTeX, big fonts, saved) ----------------
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
set(groot,'defaultTextInterpreter','latex');
set(groot,'defaultAxesFontSize',18);
set(groot,'defaultTextFontSize',22);

fig = figure('Color','w','Position',[70 70 1800 700]);

subplot(2,3,1);
imagesc(xp,yp,sdfF(XX,YY)); axis image xy; colorbar;
title('Numerical SDF (positive inside)');
xlabel('$x$'); ylabel('$y$'); hold on;
plot(X_bc(:,2), X_bc(:,1), 'k.', 'MarkerSize',4); hold off;

subplot(2,3,2);
scatter(X_pde(:,1),X_pde(:,2),8,wPDE_vec,'filled');
axis equal tight; colorbar;
title('PDE row weight $w_{\mathrm{PDE}}(x,y)$');
xlabel('$x$'); ylabel('$y$'); grid on;

subplot(2,3,3);
plot(X_bc(:,1),X_bc(:,2), 'k.'); axis equal tight; grid on;
title(sprintf('Boundary points ($N_{b}=%d$)',Nb_side));
xlabel('$x$'); ylabel('$y$');

subplot(2,3,4);
surf(XX,YY,U_pred,'EdgeColor','none'); view(2); colorbar; grid on;
title('KAPI--ELM solution'); xlabel('$x$'); ylabel('$y$');

subplot(2,3,5);
surf(XX,YY,U_exact,'EdgeColor','none'); view(2); colorbar; grid on;
title('Exact solution'); xlabel('$x$'); ylabel('$y$');

subplot(2,3,6);
surf(XX,YY,abs_err,'EdgeColor','none'); view(2); colorbar; grid on;
title(sprintf('Absolute error (MSE = %.2e)',mse));
xlabel('$x$'); ylabel('$y$');

% ---- High-resolution figure save ----
exportgraphics(fig,'TC_05_Irregular_SDF.png','Resolution',300);
disp('Saved figure: TC_05_Irregular_SDF.png');

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
exportgraphics(fig_error, 'TC_05_Irregular_SDF_error.png', 'Resolution', 300);

% (Optional) Close the temporary figure after saving
close(fig_error);
%% ===================== Helpers =====================

function [xg, yg, sdf, inMask] = build_irregular_domain_SDF(N)
% Build a "flower" shape in [0,1]^2 on an N×N grid and compute numerical SDF.
% Positive inside, 0 on boundary, negative outside.
    xg = linspace(0,1,N);
    yg = linspace(0,1,N);
    [X,Y] = meshgrid(xg, yg);

    % Flower in polar coords wrt center (0.5,0.5)
    cx=0.5; cy=0.5;
    Xc = X - cx;  Yc = Y - cy;
    R  = hypot(Xc, Yc);
    TH = atan2(Yc, Xc);

    % Radius with 5 petals
    R0 = 0.33;  A = 0.10;  nPetal = 5;
    Rb = R0 + A*cos(nPetal*TH);

    inMask = (R <= Rb);                    % inside=1, outside=0  (logical)

    % Numerical signed distance (grid spacing h)
    h  = 1/(N-1);
    D_out = bwdist(inMask);                % distance (pixels) to nearest inside
    D_in  = bwdist(~inMask);               % distance to nearest outside
    sdf = (D_in - D_out) * h;              % signed distance in physical units

    % enforce double types (contourc/bwdist friendly)
    xg    = double(xg);
    yg    = double(yg);
    sdf   = double(sdf);
    inMask = logical(inMask);
end

function XY = contourc_to_points(C)
% Convert MATLAB contourc output (for a single level) to an Nx2 point list.
% If multiple segments exist, concatenates them.
    XY = [];
    k = 1;
    while k < size(C,2)
        len = C(2,k);
        P = C(:, k+1:k+len).';
        XY = [XY; P]; %#ok<AGROW>
        k = k + len + 1;
    end
    if ~isempty(XY) && norm(XY(1,:)-XY(end,:)) < 1e-12
        XY(end,:) = [];
    end
end

function XY = bw_to_points(BW, xg, yg)
% Convert bwboundaries result to a single Nx2 polyline in (x,y)
    C = bwboundaries(BW, 'noholes');
    if isempty(C), XY = []; return; end
    [~,iMax] = max(cellfun(@(c) size(c,1), C));  % pick longest boundary
    B = C{iMax};  % [row, col]
    rr = B(:,1); cc = B(:,2);
    rr = max(1, min(rr, numel(yg)));             % clamp
    cc = max(1, min(cc, numel(xg)));
    x = xg(cc); y = yg(rr);
    XY = [x,y];
end

function XYs = resample_polyline_closed(XY, M)
% Uniformly resample a (possibly multi-segment) closed polyline into M points.
    if isempty(XY), XYs = XY; return; end
    if norm(XY(1,:) - XY(end,:)) > 1e-12         % close if not closed
        XY = [XY; XY(1,:)];
    end
    seg = sqrt(sum(diff(XY,1,1).^2,2));
    s   = [0; cumsum(seg)];
    L   = s(end);
    s_new = linspace(0, L, M+1).'; s_new(end) = [];    % remove duplicate endpoint
    XYs = [interp1(s, XY(:,1), s_new, 'linear'), ...
           interp1(s, XY(:,2), s_new, 'linear')];
end

function [XY_part, sigma_part, XY_global, sigma_global] = ...
    SAMP_POINTS_SIGMA_2D(Nx, Ny, lensX, lensY, kSigma)
% Centers & isotropic σ for partition- and global-grids on [0,1]^2.
% σ_part(i,j) = kSigma * sqrt((lensX_i/Nx)*(lensY_j/Ny)),  σ_global = const.

    if nargin < 5 || isempty(kSigma), kSigma = 1.0; end
    lensX = lensX(:).'; lensY = lensY(:).';
    if isempty(lensX) || any(lensX <= 0), error('lensX must be positive and non-empty.'); end
    if isempty(lensY) || any(lensY <= 0), error('lensY must be positive and non-empty.'); end
    sx = sum(lensX); sy = sum(lensY);
    if abs(sx - 1) > 1e-12, lensX = lensX / sx; end
    if abs(sy - 1) > 1e-12, lensY = lensY / sy; end
    kx = numel(lensX); ky = numel(lensY);

    edgesX = [0, cumsum(lensX)]; edgesX(end) = 1;
    edgesY = [0, cumsum(lensY)]; edgesY(end) = 1;

    [x_axis, x_part_idx] = build_axis_with_idx(edgesX, Nx);
    [y_axis, y_part_idx] = build_axis_with_idx(edgesY, Ny);

    [Xp, Yp] = meshgrid(x_axis, y_axis);
    XY_part = [Xp(:), Yp(:)];

    [IX, IY] = meshgrid(x_part_idx, y_part_idx);
    IX = IX(:); IY = IY(:);
    dx_local = lensX(IX) / Nx;
    dy_local = lensY(IY) / Ny;
    sigma_part = kSigma * sqrt(dx_local .* dy_local);
    sigma_part = sigma_part(:);

    xg = linspace(0, 1, kx*Nx + 2); xg = xg(2:end-1);
    yg = linspace(0, 1, ky*Ny + 2); yg = yg(2:end-1);
    [Xg, Yg] = meshgrid(xg, yg);
    XY_global = [Xg(:), Yg(:)];

    dxg = 1 / (kx * Nx);
    dyg = 1 / (ky * Ny);
    sigma_global = kSigma * sqrt(dxg * dyg) * ones(size(XY_global,1), 1);
    sigma_global = sigma_global(:);
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
