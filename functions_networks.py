import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def factorize(n):
    t=np.arange(2,n,1)
    
    
    return t[n%t==0]

def initial_values_optimization_factorial_networks(n):
    factors=factorize(n)
    sz_factors=np.size(factors)
    initial_var=int(np.floor(sz_factors/2))
    if sz_factors%2==0:
        var_1=initial_var
        var_2=var_1-1
    else:
        var_1=initial_var
        var_2=var_1
    return var_1,var_2,sz_factors,factors

def optimize_caveman(n, objective_density):
    [var_1,var_2,sz_factors,factors]=initial_values_optimization_factorial_networks(n)
    density_eval=np.zeros((sz_factors,4))
    for i in range(sz_factors):
        graph = nx.connected_caveman_graph(factors[var_1], factors[var_2])
        dens=nx.density(graph)
        density_eval[i,0]=factors[var_1]
        density_eval[i,1]=factors[var_2]
        density_eval[i,2]=dens
        if var_1==sz_factors-1:
            var_1=0
        else:
            var_1=var_1+1
        var_2=var_2-1
        density_eval[:,3]=(density_eval[:,2]-objective_density)**2
        min_caveman=np.argmin(density_eval[:,3])
    return density_eval, min_caveman

def optimize_windmill(n, objective_density):
    [var_1,var_2,sz_factors,factors]=initial_values_optimization_factorial_networks(n)
    density_eval=np.zeros((sz_factors,4))
    for i in range(sz_factors):
        graph = nx.windmill_graph(factors[var_1], factors[var_2])
        dens=nx.density(graph)
        density_eval[i,0]=factors[var_1]
        density_eval[i,1]=factors[var_2]
        density_eval[i,2]=dens
        if var_1==sz_factors-1:
            var_1=0
        else:
            var_1=var_1+1
        var_2=var_2-1
        density_eval[:,3]=(density_eval[:,2]-objective_density)**2
        min_windmill=np.argmin(density_eval[:,3])
    return density_eval, min_windmill

def optimize_smallworld(n, p, implementations, objective_density):
    factors=factorize(n)
    sz_factors=np.size(factors)
    density_eval_3=np.zeros((sz_factors,3,implementations))
    density_eval=np.zeros((sz_factors,3))
    for i in range(sz_factors):
        for k in range(implementations):
            graph = nx.watts_strogatz_graph(n, factors[int(i)], p)
            dens=nx.density(graph)
            density_eval_3[i,0,k]=factors[int(i)]
            density_eval_3[i,1,k]=dens
    density_eval_3[:,2]=(density_eval_3[:,1]-objective_density)**2
    for i in range(sz_factors):
        density_eval[i,0]=np.mean(density_eval_3[i,0,:],axis=0)
        density_eval[i,1]=np.mean(density_eval_3[i,1,:],axis=0)
        density_eval[i,2]=np.mean(density_eval_3[i,2,:],axis=0)
    min_smallworld=np.argmin(density_eval[:,2])
    return density_eval, min_smallworld

def optimize_barabassi(n,edges_barabasi_2, range_anisotropy,range_p,implementations,objective_density):
    param_retriev_i_3=np.zeros(((range_p,range_anisotropy,implementations)))
    param_retriev_j_3=np.zeros(((range_p,range_anisotropy,implementations)))
    param_retriev_i=np.zeros((range_p,range_anisotropy))
    param_retriev_j=np.zeros((range_p,range_anisotropy))
    density_eval_3=np.zeros(((range_p,range_anisotropy,implementations)))
    density_eval=np.zeros((range_p,range_anisotropy))
    for i in range(range_p):
        for j in range(range_anisotropy):
            for k in range(implementations):
                edges_barabasi_1=edges_barabasi_2/(j+1)
                graph=nx.dual_barabasi_albert_graph(n,int(n*edges_barabasi_1),int(n*edges_barabasi_2),i/range_p)
                dens=nx.density(graph)
                density_eval_3[i,j,k]=dens
                param_retriev_i_3[i,j,k]=i/range_p
                param_retriev_j_3[i,j,k]=int(n*edges_barabasi_1)
    density_eval_3[:,3]=(density_eval_3[:,2]-objective_density)**2
    for i in range(range_p):
        for j in range(range_anisotropy):
            density_eval[i,j]=np.mean(density_eval_3[i,j,:],axis=0)
            param_retriev_i[i,j]=np.mean(param_retriev_i_3[i,j,:],axis=0)
            param_retriev_j[i,j]=np.mean(param_retriev_j_3[i,j,:],axis=0)
    min_barabassi_p=np.argmin(density_eval)
    return density_eval, param_retriev_i, param_retriev_j, min_barabassi_p

def find_parameters_density(n,p_stochaisc_smallworld,edges_barabassi, range_anisotropy, range_p, implementations, objective_density):
    [density_eval_caveman, min_caveman]=optimize_caveman(n,objective_density)
    l_caveman=int(density_eval_caveman[min_caveman,0])
    k_caveman=int(density_eval_caveman[min_caveman,1])
    [density_eval_windmill, min_windmill]=optimize_windmill(n,objective_density)
    l_windmill=int(density_eval_windmill[min_caveman,0])
    k_windmill=int(int(density_eval_windmill[min_windmill,1]+1))
    [density_eval_smallworld, min_smallworld]=optimize_smallworld(n,p_stochaisc_smallworld,implementations, objective_density)
    k_smallworld=int(density_eval_smallworld[min_smallworld,0])
    [density_eval, param_retriev_i, param_retriev_j, min_barabassi_p]=optimize_barabassi(n,edges_barabassi, range_anisotropy,range_p,implementations,objective_density)
    k_barabassi=int(np.ndarray.flatten(param_retriev_j)[min_barabassi_p])
    l_barabassi=int(n*edges_barabassi)
    p_barabassi=np.ndarray.flatten(param_retriev_i)[min_barabassi_p]
    return l_caveman, k_caveman, l_windmill, k_windmill, k_smallworld, k_barabassi, l_barabassi, p_barabassi
    

def network_stats(graph):
    dens=nx.density(graph)
    clust=nx.average_clustering(graph, nodes=None, weight=None, count_zeros=True)
    degree_sequence = sorted((d for n, d in graph.degree()), reverse=True)
    dmax = max(degree_sequence)

    fig = plt.figure("Degree of a random graph", figsize=(8, 8))
    # Create a gridspec for adding subplots of different sizes
    axgrid = fig.add_gridspec(2)

    ax0 = fig.add_subplot(axgrid[0])
    Gcc = graph.subgraph(sorted(nx.connected_components(graph), key=len, reverse=True)[0])
    pos = nx.kamada_kawai_layout(Gcc)
    nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20)
    nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4)
    ax0.set_title("Connected components of G")
    ax0.set_axis_off()

    ax1 = fig.add_subplot(axgrid[1])
    ax1.plot(degree_sequence, "b-", marker="o")
    ax1.set_title("Degree Rank Plot")
    ax1.set_ylabel("Degree")
    ax1.set_xlabel("Rank")


    fig.tight_layout()
    
    
    return plt.show(), print("density",dens,"clustering coeff", clust)



