import mpi4py.MPI as mpi
import numpy as np
import pypacmensl.state_set.constrained as sp
import matplotlib.pyplot as plt

def plot_state_set(comm, my_set):
    """Generate plots of the state space given StateSetConstrained object."""
    X = my_set.GetStates()
    num_procs = comm.size
    my_rank = comm.rank

    X_list = [X]
    if my_rank == 0 :
        for i in range(1, num_procs):
            X_list.append(comm.recv(source=i, tag=1))
    else:
        comm.send(X, dest=0, tag=1)

    # Let's plot!
    colors = ['red', 'green', 'blue', 'magenta']
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    leg = []
    if my_rank == 0:
        for i in range(0, num_procs):
            ax.scatter(X_list[i][:,0], X_list[i][:,1],
                        c = colors[i],
                       # c=[[1-0.3*i, 1-0.1*i, 0.25*i + 0.1]]
                       )
            leg.append('Processor ' + str(i))
        ax.legend(leg)
        ax.set_xlabel('Species 1')
        ax.set_ylabel('Species 2')
    return fig, ax

#%%
comm = mpi.COMM_WORLD
num_procs = comm.size
my_rank = comm.rank

sm = np.array([[1,0],[-1,0], [0,1], [0,-1]], dtype=np.intc)
x0 = np.array([[1,2],[0,0]], dtype=np.intc)
def simple_constr(X, out):
    # The spear of Adun
    n_constr = 5
    out[:,0] = X[:,0]
    out[:,1] = X[:,1]
    out[:,2] = X[:,0] + X[:,1]
    out[:,3] = -0.5*X[:,0] + X[:,1]
    out[:,4] = X[:,0] - 0.5*X[:,1]

bounds = np.array([300, 300, 300, 70, 70])
#%%
my_set_block = sp.StateSetConstrained(comm)
my_set_block.SetLBMethod("block")
my_set_block.SetStoichiometry(sm)
my_set_block.SetConstraint(simple_constr, bounds)
my_set_block.SetUp()
my_set_block.AddStates(x0)
my_set_block.Expand()
f0, ax0 = plot_state_set(comm, my_set_block)
ax0.set_title('Naively partitioned FSP')
#%%
my_set_graph = sp.StateSetConstrained(comm)
my_set_graph.SetStoichiometry(sm)
my_set_graph.SetLBMethod("graph")
my_set_graph.SetConstraint(simple_constr, bounds)
my_set_graph.SetUp()
my_set_graph.AddStates(x0)
my_set_graph.Expand()
f1 , ax1 = plot_state_set(comm, my_set_graph)
ax1.set_title('Graph-partitioned FSP')
#%%
my_set_hypergraph = sp.StateSetConstrained(comm)
my_set_hypergraph.SetStoichiometry(sm)
my_set_hypergraph.SetLBMethod("hypergraph")
my_set_hypergraph.SetConstraint(simple_constr, bounds)
my_set_hypergraph.SetUp()
my_set_hypergraph.AddStates(x0)
my_set_hypergraph.Expand()
f2, ax2 = plot_state_set(comm, my_set_hypergraph)
ax2.set_title('Hypergraph-partitioned FSP')

#%%
if (my_rank == 0):
    f0.savefig('naive_fsp.eps', format='eps')
    f1.savefig('graph_fsp.eps', format='eps')
    f2.savefig('hypergraph_fsp.eps', format='eps')
    plt.show()
