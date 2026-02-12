import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from matplotlib.colors import BoundaryNorm, TwoSlopeNorm
import os
from scipy.optimize import curve_fit
from sympy import gamma


def compute_edges(arr):
    arr = np.asarray(arr)
    edges = (arr[:-1] + arr[1:]) / 2
    edges = np.concatenate([[arr[0] - (edges[0] - arr[0])], edges, [arr[-1] + (arr[-1] - edges[-1])]])
    return edges

def decide_cbar_label(metric):
    cbar_label = fr"$\tau_{metric[-1]}$ at time $t=200$"
    return cbar_label

def plot_cont_phase_diagram(df, metric, plot_name, output_dir):    

    pivot = df.pivot(index="S", columns="F", values=metric)
    pivot.index = pivot.index.astype(float)
    pivot.columns = pivot.columns.astype(float)

    pivot = pivot.sort_index(axis=0, ascending=False).sort_index(axis=1)

    plt.figure(figsize=(14, 10))
    fig, ax = plt.subplots(figsize=(14, 10))

    # Convert pivot to numpy
    Z = pivot.values
    x = pivot.columns.values
    y = pivot.index.values
    x_edges = compute_edges(x)
    y_edges = compute_edges(y)

    if "0p24" in plot_name.name:
        v_min = 0.24
        barticks = [0.25 + i*0.05 for i in range(16)]
    else:
        v_min = 0.0
        barticks = [v_min + i*0.1 for i in range(11)]

    mesh = ax.pcolormesh(
        x, y, Z,
        cmap="coolwarm",
        vmin=v_min,
        vmax=1.0,
    )

    for xv in x_edges:
        ax.axvline(xv, color="gray", linestyle="--", linewidth=1)
    for yv in y_edges:
        ax.axhline(yv, color="gray", linestyle="--", linewidth=1)

    cbar = fig.colorbar(mesh, ax=ax, ticks=barticks)
    cbar.set_label(decide_cbar_label(metric), fontsize=28)
    cbar.ax.tick_params(labelsize=24)
    ax.set_yscale('log')
    # ax.set_xscale('log')
    # plt.title(fr"Heatmap of final belief $\tau_{metric[-1]}$")
    plt.xlabel(r"Flexibility $\phi$", fontsize=28)
    plt.ylabel(r"Confirmation bias $s$", fontsize=28)
    plt.yticks(fontsize=24)
    plt.xticks(np.arange(0.05, 1.0, step=0.1), fontsize=24, rotation=45)
    plt.tight_layout()
    out_path = output_dir / plot_name.name
    plt.savefig(out_path, dpi=400)
    plt.show()
    plt.close()

    return None


def plot_discrete_phase_diagram(df, metric, plot_name, output_dir):    

    if "0p24" in plot_name.name:
        vmin = 0.24
        barticks = [0.25 + i*0.05 for i in range(16)]
    else:
        vmin = 0.0
        barticks = [vmin + i*0.1 for i in range(11)]
    vmax = 1.0
    n_bins = 19
    bounds = np.linspace(vmin, vmax, n_bins + 1)
    cmap = cm.get_cmap("coolwarm", n_bins)
    norm = BoundaryNorm(boundaries=bounds, ncolors=cmap.N)

    pivot = df.pivot(index="S", columns="F", values=metric)
    pivot.index = pivot.index.astype(float)
    pivot.columns = pivot.columns.astype(float)

    pivot = pivot.sort_index(axis=0, ascending=False).sort_index(axis=1)

    plt.figure(figsize=(14, 10))
    fig, ax = plt.subplots(figsize=(14, 10))

    # Convert pivot to numpy
    Z = pivot.values
    x = pivot.columns.values
    y = pivot.index.values
    x_edges = compute_edges(x)
    y_edges = compute_edges(y)

    mesh = ax.pcolormesh(
        x, y, Z,
        cmap=cmap,
        norm=norm,
        shading="nearest"
    )

    for xv in x_edges:
        ax.axvline(xv, color="gray", linestyle="--", linewidth=1)
    for yv in y_edges:
        ax.axhline(yv, color="gray", linestyle="--", linewidth=1)

    cbar = fig.colorbar(mesh, ax=ax, ticks=barticks)
    cbar.set_label(decide_cbar_label(metric), fontsize=28)
    cbar.ax.tick_params(labelsize=24)
    ax.set_yscale('log')
    # ax.set_xscale('log')
    # plt.title(fr"Heatmap of final belief $\tau_{metric[-1]}$")
    plt.xlabel(r"Flexibility $\phi$", fontsize=28)
    plt.ylabel(r"Confirmation bias $s$", fontsize=28)
    plt.yticks(fontsize=24)
    plt.xticks(np.arange(0.05, 1.0, step=0.1), fontsize=24, rotation=45)
    plt.tight_layout()
    out_path = output_dir / plot_name
    plt.savefig(out_path, dpi=400)
    plt.show()
    plt.close()

    return None

def plot_differences(df_SQUARE, df_BA, metric, plot_name, output_dir):

    pivot_SQUARE = df_SQUARE.pivot(index="S", columns="F", values=metric)
    pivot_SQUARE.index = pivot_SQUARE.index.astype(float)
    pivot_SQUARE.columns = pivot_SQUARE.columns.astype(float)
    pivot_BA = df_BA.pivot(index="S", columns="F", values=metric)
    pivot_BA.index = pivot_BA.index.astype(float)
    pivot_BA.columns = pivot_BA.columns.astype(float)

    pivot_SQUARE = pivot_SQUARE.sort_index(axis=0, ascending=False).sort_index(axis=1)
    pivot_BA = pivot_BA.sort_index(axis=0, ascending=False).sort_index(axis=1)

    plt.figure(figsize=(14, 10))
    fig, ax = plt.subplots(figsize=(14, 10))

    # Convert pivot to numpy
    Z = pivot_SQUARE.values - pivot_BA.values
    x = pivot_SQUARE.columns.values
    y = pivot_SQUARE.index.values
    x_edges = compute_edges(x)
    y_edges = compute_edges(y)

    vmax = np.nanmax(np.abs(Z))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    mesh = ax.pcolormesh(
        x, y, Z,
        cmap="coolwarm",
        norm=norm,
    )

    for xv in x_edges:
        ax.axvline(xv, color="gray", linestyle="--", linewidth=1.0)
    for yv in y_edges:
        ax.axhline(yv, color="gray", linestyle="--", linewidth=1.0)

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Difference in " + decide_cbar_label(metric).lower(), fontsize=28)
    cbar.ax.tick_params(labelsize=24)
    ax.set_yscale('log')
    plt.xlabel(r"Flexibility $\phi$", fontsize=28)
    plt.ylabel(r"Confirmation bias $s$", fontsize=28)
    plt.yticks(fontsize=24)
    plt.xticks(np.arange(0.05, 1.0, step=0.1), fontsize=24, rotation=45)
    plt.tight_layout()
    out_path = output_dir / f"cont_{plot_name.name}"
    plt.savefig(out_path, dpi=400)
    plt.show()
    plt.close()

    plt.figure(figsize=(14, 10))
    fig, ax = plt.subplots(figsize=(14, 10))

    # Convert pivot to numpy
    Z = pivot_SQUARE.values - pivot_BA.values
    x = pivot_SQUARE.columns.values
    y = pivot_SQUARE.index.values
    x_edges = compute_edges(x)
    y_edges = compute_edges(y)

    vmax = np.nanmax(np.abs(Z))
    n_bins = 5
    if np.nanmin(Z) < -1e-2:
        bounds = [np.nanmin(Z), -1e-2, -1e-3, 1e-3, 1e-2, vmax]
    else:
        bounds = [np.nanmin(Z), -5e-3, -1e-3, 1e-3, 5e-3, vmax]
    cmap = cm.get_cmap("coolwarm", n_bins)
    norm = BoundaryNorm(boundaries=bounds, ncolors=cmap.N)

    mesh = ax.pcolormesh(
        x, y, Z,
        cmap=cmap,
        norm=norm,
        shading="nearest"
    )

    for xv in x_edges:
        ax.axvline(xv, color="gray", linestyle="--", linewidth=1.0)
    for yv in y_edges:
        ax.axhline(yv, color="gray", linestyle="--", linewidth=1.0)

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Difference in " + decide_cbar_label(metric).lower(), fontsize=28)
    cbar.ax.tick_params(labelsize=24)
    ax.set_yscale('log')
    plt.xlabel(r"Flexibility $\phi$", fontsize=28)
    plt.ylabel(r"Confirmation bias $s$", fontsize=28)
    plt.yticks(fontsize=24)
    plt.xticks(np.arange(0.05, 1.0, step=0.1), fontsize=24, rotation=45)
    plt.tight_layout()
    out_path = output_dir / f"discrete_{plot_name.name}"
    plt.savefig(out_path, dpi=400)
    plt.show()
    plt.close()

    return None

def plot_critical(df, fs, ss, metric, plot_name, output_dir):

    pivot = df.pivot(index="S", columns="F", values=metric)
    pivot.index = pivot.index.astype(float)
    pivot.columns = pivot.columns.astype(float)

    pivot = pivot.sort_index(axis=0, ascending=False).sort_index(axis=1)

    plt.figure(figsize=(14, 10))
    fig, ax = plt.subplots(figsize=(14, 10))

    # Convert pivot to numpy
    Z = pivot.values
    x = pivot.columns.values
    y = pivot.index.values
    x_edges = compute_edges(x)
    y_edges = compute_edges(y)

    vmin = np.nanmin(Z)
    vmax = np.nanmax(Z)

    mesh = ax.pcolormesh(
    x, y, Z,
    cmap="coolwarm",
    vmax=vmax
    )

    for xv in x_edges:
        ax.axvline(xv, color="gray", linestyle="--", linewidth=1.0)
    for yv in y_edges:
        ax.axhline(yv, color="gray", linestyle="--", linewidth=1.0)

    bounds = np.linspace(vmin, vmax, 9)
    # bounds = np.append(bounds, vmax)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_ticks(bounds)
    cbar.set_ticklabels([f"{b:.5f}" for b in bounds], fontsize=34)
    cbar.set_label(decide_cbar_label(metric), fontsize=44)
    cbar.ax.tick_params(labelsize=26)

    ax.set_yscale('log')
    ax.set_xscale('log')
    # plt.title(fr"Heatmap of final Truthfulness $\tau_3$")
    plt.xlabel(r"Flexibility $\phi$", fontsize=44)
    plt.ylabel(r"Confirmation bias $s$", fontsize=44)
    ax.set_yticks(ss)
    ax.set_yticklabels([f"{v:.0e}" for v in ss], fontsize=34) 
    ax.set_xticks(fs)
    if np.min(fs) <0.1:
        ax.set_xticklabels([f"{v:.0e}" for v in fs], fontsize=34)
    else:
        ax.set_xticklabels([v for v in fs], fontsize=34)
    ax.tick_params(axis='x', pad=12)
    ax.tick_params(axis='y', pad=12)
    fig.tight_layout()
    out_path = output_dir / plot_name
    plt.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.show()
    plt.close()

    return None

LABELSIZE = 28
TICKSIZE = 20
LEGENDSIZE = 20


def plot_phase_line(df, variable, metric, plot_name, output_dir, x_type, y_type, inverse):

    tau = df[metric]
    std = metric + "_std"
    SEM = df[std]/np.sqrt(200)

    if variable == "F":
        if inverse:
            var = 1/df[variable]
            xlabel = r"Flexibility $1/\phi$"
        else:
            var = df[variable]
            xlabel = r"Flexibility $\phi$"
    elif variable == "S":
        if inverse:
            var = 1/df[variable]
            xlabel = r"Confirmation bias $1/s$"
        else:
            var = df[variable]
            xlabel = r"Confirmation bias $s$"

    tau = df[metric]
    if metric == "T3":
        tau = tau - 0.25
        labelmetric = r"$\tau_3 - 0.25$"
    else:
        labelmetric = fr"$\tau_{metric[-1]}$"
    
    plt.figure(figsize=(9, 6))
    # plt.scatter(var, df[metric], marker="o")
    plt.errorbar(var, tau, SEM, fmt='o', capsize=4, elinewidth=1, label=labelmetric)
    plt.errorbar(var, 1-df[metric], SEM, fmt='o', capsize=4, elinewidth=1, label=fr"$1.0 - \tau_{metric[-1]}$")
    plt.xlabel(xlabel, fontsize=LABELSIZE)
    # plt.xlim((0.01, 1))
    if x_type == "log":
        plt.xscale("log")
    if y_type == "log":
        plt.yscale("log")
    plt.ylabel(decide_cbar_label(metric), fontsize=LABELSIZE)
    plt.yticks(fontsize=TICKSIZE)
    plt.xticks(fontsize=TICKSIZE)
    # plt.title(fr"Final {metric} vs $\phi$ (s={s_val})")
    plt.grid(True)
    plt.legend(fontsize=LEGENDSIZE)
    plt.tight_layout()
    out_path = output_dir / plot_name
    plt.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.show()
    plt.close()

    return None

def plot_phase_derivatives(dfs, variable, metric, Ns, plot_name, output_dir):

    if variable == "F":
        xlabel = r"Flexibility $\phi$"
        dylabel = fr"$d\,\tau_{{{metric[-1]}}} / d\phi$"
        d2ylabel = fr"$d^2\,\tau_{{{metric[-1]}}} / d\phi^2$"
        var_ylabel = fr"$N\cdot\mathrm{{Var}}(\tau_{{{metric[-1]}}})$"
    elif variable == "S":
        xlabel = r"Confirmation bias $s$"
        dylabel = fr"$d\,\tau_{{{metric[-1]}}} / ds$"
        d2ylabel = fr"$d^2\,\tau_{{{metric[-1]}}} / ds^2$"
        var_ylabel = fr"$N\cdot\mathrm{{Var}}(\tau_{{{metric[-1]}}})$"

    # f_all = (df["F"].values - Fc) / Fc

    fig1, ax1 = plt.subplots(figsize=(9, 6))  # d tau / d var
    fig2, ax2 = plt.subplots(figsize=(9, 6))  # d2 tau / d var2
    fig3, ax3 = plt.subplots(figsize=(9, 6))  # Var(tau)

    for i, N in enumerate(Ns):

        # df = dfs[i].sort_values(variable)  # safer for gradients
        df = dfs[i]
        tau = df[metric]
        variance = df[metric + "_std"]**2

        if variable == "F":
            df = df[df["F"] > 5e-5]     # only F > 10^{-3}

        var = df[variable].to_numpy()
        tau = df[metric].to_numpy()
        variance = (df[f"{metric}_std"].to_numpy()) ** 2

        # If var not strictly increasing, gradients can be weird; sorting helps.
        d1 = np.gradient(tau, var)
        d2 = np.gradient(d1, var)

        if N == 256:
            color = "tab:blue"
        elif N == 1024:
            color = "orange"
        elif N == 4096:
            color = "green"

        # --- plot each N on same axes ---
        ax1.plot(var, d1, label=fr"$N={N}$", color=color)
        ax2.plot(var, d2, label=fr"$N={N}$", color=color)
        ax3.plot(var, variance*N, label=fr"$N={N}$, $N\cdot\mathrm{{Var}}(\tau_{{{metric[-1]}}})$", color=color)

        # Optionally mark peak variance point for each N:
        peak_i = int(np.argmax(variance))

        if variable == "F":
            inflection_label = fr"$\phi_\mathrm{{inflection}} = {var[peak_i]:.5f}$"
        elif variable == "S":
            inflection_label = fr"$s_\mathrm{{inflection}} = {var[peak_i]:.5f}$"
        ax3.axvline(var[peak_i], linestyle="--", label=inflection_label, linewidth=1.0, color= color, alpha=0.8)

            # ---- style/labels ----
    for ax in (ax1, ax2, ax3):
        ax.set_xlabel(xlabel, fontsize=LABELSIZE)
        ax.tick_params(labelsize=TICKSIZE)
        ax.grid(True)
        ax.set_xscale("log")

        ax.legend(fontsize=LEGENDSIZE)

    ax1.set_ylabel(dylabel, fontsize=LABELSIZE)
    ax2.set_ylabel(d2ylabel, fontsize=LABELSIZE)
    # ax2.set_yscale("log")
    ax2.axhline(0, color="k", linewidth=0.8)

    ax3.set_ylabel(var_ylabel, fontsize=LABELSIZE)
    ax3.set_yscale("log")

    Ns_tag = "-".join(map(str, Ns))
    f1 = os.path.join(output_dir, f"{plot_name}_d1_{variable}_{metric}_Ns{Ns_tag}.png")
    f2 = os.path.join(output_dir, f"{plot_name}_d2_{variable}_{metric}_Ns{Ns_tag}.png")
    f3 = os.path.join(output_dir, f"{plot_name}_var_{variable}_{metric}_Ns{Ns_tag}.png")

    fig1.tight_layout(); fig1.savefig(f1, dpi=400, bbox_inches='tight')
    fig2.tight_layout(); fig2.savefig(f2, dpi=400, bbox_inches='tight')
    fig3.tight_layout(); fig3.savefig(f3, dpi=400, bbox_inches='tight')
    plt.show()
    plt.close(fig1); plt.close(fig2); plt.close(fig3)

    return None

def sigmoid(F, a, Fc, Tlow, Thigh):
    return Tlow + (Thigh - Tlow) / (1 + np.exp(-a*(F - Fc)))

def powerlaw(F, gamma, Fc, Tlow, Thigh):
    x = np.maximum(F - Fc, 0.0)
    return Tlow + (Thigh - Tlow) * (x)**gamma

def pure_powerlaw(x, A, gamma, xc):
    return A * (x/xc)**gamma


def plot_power_law(df, variable, metric, plot_name, output_dir):
    
    tau = df[metric]
    std = metric + "_std"
    SEM = df[std]/np.sqrt(200)
    Z = np.abs(tau - 0.25)/SEM

    if variable == "F":
        var = df[variable]
        xlabel = r"Flexibility $\phi$"
        labelmetric2 = r"$\phi$ where $Z<2$"
        labelmetric1 = r"$\phi$ where $Z<1$"
        inflection_label = fr"$\phi_\mathrm{{inflection}}={var[np.argmax(df['T3_std']**2)]:.5f}$"
    elif variable == "S":
        var = df[variable]
        xlabel = r"Confirmation bias $s$"
        labelmetric2 = r"$s$ where $Z<2$"
        labelmetric1 = r"$s$ where $Z<1$"
        inflection_label = fr"$s_\mathrm{{inflection}}={var[np.argmax(df['T3_std']**2)]:.5f}$"

    p0 = [2.0, 0.1, np.min(tau), np.max(tau)] 
    bounds = (
        [0.0001, 1e-6, -np.inf, -np.inf],
        [1000.0, 1.0, np.inf, np.inf]
    )

    params, cov = curve_fit(sigmoid, var, tau, p0=p0, bounds=bounds, maxfev=50000)
    varpower = var[8:40]
    Tpower = tau[8:40]-0.25
    # powerparams, powercov = curve_fit(pure_powerlaw, varpower, Tpower, p0=[1e-3, 1.0, np.median(varpower)],
    #                    bounds=([0, -np.inf, 0], [np.inf, np.inf, np.inf]))
    a, varc, Tlow, Thigh = params
    # gamma, varc_power, Tlow_power, Thigh_power = powerparams
    # gamma = powerparams[1]

    var_sorted = np.linspace(np.min(var), np.max(var), 500)
    tau_fit_curve = sigmoid(var_sorted, *params)

    # var_power_grid = np.logspace(np.log10(varpower.min()), np.log10(varpower.max()), 300)
    # tau_power_fit = pure_powerlaw(var_power_grid, *powerparams)
    gamma, log10C = np.polyfit(np.log10(varpower), np.log10(Tpower), 1)
    C = 10**log10C
    xg = np.logspace(np.log10(varpower.min()), np.log10(varpower.max()), 300)
    yg = C * xg**gamma

    ly_fit = gamma * np.log10(varpower) + log10C
    SS_res = np.sum((np.log10(Tpower) - ly_fit)**2)
    SS_tot = np.sum((np.log10(Tpower) - np.mean(np.log10(Tpower)))**2)
    R2_gamma = 1 - SS_res / SS_tot
    print(f"Power-law R^2 (log-log) = {R2_gamma:.4f}")

    tau_fit = sigmoid(var, *params)
    SS_res = np.sum((tau - tau_fit)**2)
    SS_tot = np.sum((tau - np.mean(tau))**2)
    R2_sigmoid = 1 - SS_res / SS_tot
    print(f"Sigmoid R^2 = {R2_sigmoid:.4f}")

    var_ref0 = 1.8e-2
    tau_ref0 = 1e-1
    gamma_test = 1.8
    var_ref = np.linspace(3e-5, var_ref0, 100)
    tau_ref = tau_ref0 * (var_ref/var_ref0)**gamma_test

    plt.figure(figsize=(9,6))
    plt.scatter(var, Z, label=r"Standardized deviation $Z$", color='green')
    plt.vlines(var[np.where(Z<2)[0][-1]], ymin=0, ymax=10000, color='orange', linestyle="dashed", label=labelmetric2)
    plt.vlines(var[np.where(Z<1)[0][-1]], ymin=0, ymax=10000, color='orange', linestyle="dashdot", label=labelmetric1)
    plt.xscale("log")
    plt.yscale("log")
    plt.grid()
    plt.ylim((0,6000))
    plt.xlabel(xlabel, fontsize=LABELSIZE)
    plt.ylabel(r"Standardized deviation $Z$", fontsize=LABELSIZE)
    plt.yticks(fontsize=TICKSIZE)
    plt.xticks(fontsize=TICKSIZE)
    plt.legend(fontsize=LEGENDSIZE, loc="upper left")
    plt.tight_layout()
    out_path = output_dir / f"{plot_name}_Z.png"
    plt.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.show()
    plt.close()

    plt.figure(figsize=(9,6))
    # plt.scatter(F[:80], T[:80]-0.25, label="data", s=30)
    # plt.scatter(F[:80], SNR, label=r"Standardized deviation $Z$", color='green')
    # plt.vlines(F[np.where(SNR<2)[0][-1]], ymin=0, ymax=10000, color='orange', linestyle="dashed", label=r"$\phi$ where SNR<2")
    # plt.vlines(F[np.where(SNR<1)[0][-1]], ymin=0, ymax=10000, color='orange', linestyle="dashdot", label=r"$\phi$ where SNR<1")
    if variable=="F":
        plt.errorbar(var[:80], (tau[:80]-0.25), SEM[:80], fmt='o', capsize=4, elinewidth=1, label=fr"$\tau_{metric[-1]}-0.25$", alpha=0.7, zorder=14)
        # plt.plot(var_ref, tau_ref, label=fr"reference line $\gamma=${gamma_test:.2f}", color='black', linestyle='--', zorder=10)
        plt.plot(xg, yg, label=fr"power-law fit $\beta_\phi=${gamma:.3f}, $R^2={R2_gamma:.3f}$", color='orange', linestyle='--', zorder=15, linewidth=3)
    else:
        plt.errorbar(var, (tau-0.25), SEM, fmt='o', capsize=4, elinewidth=1, label=fr"$\tau_{metric[-1]}-0.25$")
    # plt.plot(var_sorted, tau_fit_curve-0.25, 'r-', label="sigmoid fit")
    # plt.plot(Fpower_sorted, T_power_fit_curve-0.25, 'g--', label=fr"power-law fit $\gamma=${gamma:.2f}")#, linestyle="dashed")
    # plt.vlines(F[np.argmax(df_flex["T3_std"]**2)], ymin=0.0, ymax=1.02, color="orange", linestyles="dashed", alpha=0.8, label=r"point of inflection ($\phi_c$)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(xlabel, fontsize=LABELSIZE)
    plt.ylabel(fr"$\tau_{metric[-1]} - 0.25$ at time $t=200$", fontsize=LABELSIZE)
    plt.yticks(fontsize=TICKSIZE)
    plt.xticks(fontsize=TICKSIZE)
    # plt.title(f"Sigmoid fit, R²={R2:.3f}")
    plt.legend(fontsize=LEGENDSIZE, loc="upper left")
    # plt.ylim((0.0, 1.02))
    # plt.xlim((0, 0.355))
    plt.grid()
    plt.tight_layout()
    out_path = output_dir / f"{plot_name}_log.png"
    plt.savefig(out_path, dpi=400)
    plt.show()
    plt.close()

    plt.figure(figsize=(9,6))
    plt.scatter(var, tau, label=fr"$\tau_{metric[-1]}$", s=30)
    plt.plot(var_sorted, tau_fit_curve, 'r-', label=fr"sigmoid fit, $R^2={R2_sigmoid:.3f}$")#, linestyle="dashed")
    plt.vlines(var[np.argmax(df["T3_std"]**2)], ymin=0.0, ymax=1.02, color="orange", linestyles="dashed", alpha=0.8, label=inflection_label)
    # plt.xscale("log")
    # plt.yscale("log")
    plt.xlabel(xlabel, fontsize=LABELSIZE)
    plt.ylabel(decide_cbar_label(metric), fontsize=LABELSIZE)
    plt.yticks(fontsize=TICKSIZE)
    plt.xticks(fontsize=TICKSIZE)
    # plt.title(f"Sigmoid fit, R²={R2:.3f}")
    plt.legend(fontsize=LEGENDSIZE)
    plt.ylim((0.0, 1.02))
    plt.xlim((0, 0.355))
    plt.grid()
    plt.tight_layout()
    out_path = output_dir / f"{plot_name}_sigmoid.png"
    plt.savefig(out_path, dpi=400)
    plt.show()
    plt.close()

    return None