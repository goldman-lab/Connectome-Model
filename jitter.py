import graph_tool.all as gt
import pickle
import numpy as np
import pandas
import random
import scipy.io
import h5py
import copy
import sys

def sorted_eigs(X):
    n = np.shape(X)[0]
    y,v = np.linalg.eig(X)
    vtmp = copy.deepcopy(y)
    for i in range(n):
        for j in range(i+1,n):
            if y[j] > y[i]:
                tmp = y[i]
                y[i] = y[j]
                y[j] = tmp
                vtmp[:] = v[:,i]
                v[:,i] = v[:,j]
                v[:,j] = vtmp[:]
    return y,v

def generate_blockIds(idpos,blocks):
    N = len(idpos)
    blockIds = -np.ones(N)
    for i in range(N):
        for j in range(len(blocks)):
            if idpos[i] in blocks[j]:
                blockIds[i] = j
    return blockIds

def final_adjustments(W,cdf):
    # make DOs negative
    W[:,cdf.loc['vest']] = -W[:,cdf.loc['vest']]
    W[:,cdf.loc['MO']] = -W[:,cdf.loc['MO']]

    # set ABD,vSPNs outgoing to 0
    W[:,cdf.loc['abdm']] = 0
    W[:,cdf.loc['abdi']] = 0
    W[:,cdf.loc['vspns']] = 0

    # set IBN outgoing to 0
    W[:,cdf.loc['Ibni']] = 0
    W[:,cdf.loc['Ibnm']] = 0

    # make Axial modules 0
    W[:,cdf.loc['axl']] = 0
    W[cdf.loc['axl'],:] = 0

    W[:,cdf.loc['axlm']] = 0
    W[cdf.loc['axlm'],:] = 0
    return W

def simulate_ks(W_in,ymax,ynew=0.99,tau=0.1):
    W = copy.deepcopy(W_in)
    W = ynew*W/ymax
    N = np.shape(W)[0]
    y,v = sorted_eigs(W)
    #if np.sum(np.real(v[:,0])) < 0:
    #    v[:,0] = -v[:,0]
    #if np.sum(np.real(v[:,0])) > 0:
    #    v[:,1] = -v[:,1]
    #v_in = v[:,0] + v[:,1]
    v_in = abs(np.random.randn(N))
    v_in = 0.1*v_in + np.real(np.sum(v[:,0:1],axis=1))
    v_in += 1*np.real(np.sum(v[:,1:3],axis=1))
    v_in = v_in / np.linalg.norm(v_in)
    input_filter = 0.001*np.exp(-np.linspace(0,10,101))
    I = np.zeros(7000)
    I[995:1005] = 1e5
    I = np.convolve(I,input_filter)[0:7000]
    r = np.zeros((7000,N))
    dt = 0.001
    positions = np.ones((36,2))
    positions[12:24,0] = 2
    positions[24:,0] = 3
    responses = np.ones((36,N))
    for k in range(3):
        r[0,:] = r[-1,:]
        for i in range(1,7000):
            r[i,:] = r[i-1,:] + dt*(np.dot(W,r[i-1,:]) - r[i-1,:] + I[i-1]*v_in)/tau
            #r = r * (r > 0)
        responses[k*12:(k+1)*12,:] = r[3333::333,:]
    ks = np.zeros(N)
    for i in range(N):
        res = scipy.optimize.lsq_linear(positions,responses[:,i],bounds=([0,-np.inf],[np.inf,np.inf]))
        ks[i] = res.x[0]
    return ks

def runStochasticBlockModel(g,nb):
    SBM = gt.minimize_blockmodel_dl(g,state_args=dict(B=nb,recs=[g.ep["#synapses"]],rec_types=["discrete-poisson"]),multilevel_mcmc_args=dict(B_min=nb,B_max=nb))
    maxE = SBM.entropy()
    minE = SBM.entropy()
    new_extremum = True
    count = 0
    while new_extremum and count < 10:
        new_extremum = False
        print(count,minE)
        for i in range(1000):
            test_SBM = gt.minimize_blockmodel_dl(g,state_args=dict(B=nb,recs=[g.ep["#synapses"]],rec_types=["discrete-poisson"]),multilevel_mcmc_args=dict(B_min=nb,B_max=nb))
            if test_SBM.entropy() > maxE:
                maxE = test_SBM.entropy()
                new_extremum = True
            elif test_SBM.entropy() < minE:
                minE = test_SBM.entropy()
                SBM = test_SBM
                new_extremum = True
        count += 1
    print('Final:',minE)
    return SBM

def clusterModOSBM(nBlocks,centerMethod='SBM',customMat=None):
    test_bid = sbmid
    if centerMethod == 'Louvain':
        test_bid = lbid
    om_mask = (test_bid == 1)
    A_OM = connMat[np.ix_(om_mask,om_mask)]
    if customMat != None:
        A_OM = customMat[np.ix_(om_mask,om_mask)]
    es = A_OM.T.nonzero()
    g = gt.Graph()
    g.add_edge_list(np.transpose(es))
    ew = g.new_edge_property("int32_t")
    ew.a = A_OM.T[es] 
    g.ep['#synapses'] = ew
    SBM = runStochasticBlockModel(g,nBlocks)
    blockLabels = np.unique(SBM.get_blocks().a)
    new_sbmid = copy.deepcopy(test_bid)
    new_sbmid[om_mask] = (SBM.get_blocks().a == blockLabels[0])
    for i in range(1,nBlocks):
        new_sbmid[om_mask] += (i+1)*(SBM.get_blocks().a == blockLabels[i])
    return new_sbmid

def louvain_cluster(A,gamma):
    N = np.shape(A)[0]
    outdeg = np.sum(A,axis=0)
    indeg = np.sum(A,axis=1)
    N_s = np.sum(A)
    B = A - (gamma/N_s)*np.outer(indeg,outdeg)
    B = (B + B.T)/(2*N_s)
    H = copy.deepcopy(B)
    blockIds = np.arange(N)
    nodes = np.arange(N)
    nodeBlocks = np.arange(N)
    nblocks = N
    Q0 = N*N*np.min(B) - 1
    Q = np.trace(B)
    while Q - Q0 > 1e-10:
        flag = True
        while flag:
            flag = False
            np.random.shuffle(nodes)
            for node in nodes:
                bid = nodeBlocks[node]
                dQ = H[:,node] - H[bid,node] + B[node,node]
                dQ[bid] = 0
                
                new_bid = np.argmax(dQ)
                if dQ[new_bid] > 1e-10:
                    flag = True
                    nodeBlocks[node] = new_bid
                    H[bid,:] = H[bid,:] - B[node,:]
                    H[new_bid,:] = H[new_bid,:] + B[node,:]
        x = np.unique(nodeBlocks)
        for i in range(np.shape(x)[0]):
            nodeBlocks[nodeBlocks == x[i]] = i
        new_blockIds = np.zeros(N)
        for i in range(nblocks):
            new_blockIds[blockIds == i] = nodeBlocks[i]
        blockIds = new_blockIds
        nblocks = np.shape(x)[0]
        B1 = np.zeros((nblocks,nblocks))
        for i in range(nblocks):
            for j in range(i,nblocks):
                B1[i,j] = np.sum(B[np.ix_(nodeBlocks == i,nodeBlocks == j)])
                B1[j,i] = B1[i,j]
        B = B1
        H = copy.deepcopy(B)
        nodeBlocks = np.arange(nblocks)
        nodes = np.arange(nblocks)
        Q0 = Q
        Q = np.trace(B)
    #print(Q,nblocks)
    return Q,blockIds

def clusterCenterLouvain(gamma,customMat=None):
    centerMask = (cellIDs == '_Int_') | (cellIDs == '_DOs_') | (cellIDs == '_Axl_')
    A_center = connMat[np.ix_(centerMask,centerMask)]
    if not (None in (customMat)):
        A_center = customMat[np.ix_(centerMask,centerMask)]
    Q,center_lbid = louvain_cluster(A_center,gamma)
    maxQ = Q
    minQ = Q
    new_extremum = True
    count = 0
    while new_extremum and count < 10:
        new_extremum = False
        print(count,maxQ)
        for i in range(1000):
            Q,bid = louvain_cluster(A_center,gamma)
            if Q > maxQ:
                maxQ = Q
                new_extremum = True
                center_lbid = bid
            elif Q < minQ:
                minQ = Q
                new_extremum = True
        count += 1
    nblocks = np.shape(np.unique(center_lbid))[0]
    print('Final:',maxQ,nblocks)
    new_lbid = -np.ones(N)
    for i in range(nblocks):
        new_lbid[centerMask] += (i+1)*(center_lbid == i)
    return new_lbid

def jitter_connections(connMatrix,cdf,totalInputs):
    W = np.zeros(np.shape(connMatrix),dtype=float)
    W[:,:] = abs(connMatrix)[:,:]
    fp = 1.0 - 0.86 # false positive rate
    fn = 0.86/0.83 - 0.86 # false negative rate
    false_negatives = np.zeros(np.shape(W))
    post_blocks = [np.where(cdf.loc['integ'])[0],
                   np.where(cdf.loc['vest'])[0],
                   np.where(cdf.loc['axl'])[0],
                   np.where(cdf.loc['abdm'])[0],
                   np.where(cdf.loc['abdi'])[0]]
    pre_blocks = [np.where(cdf.loc['integ'])[0],
                  np.where(cdf.loc['vest'])[0],
                  np.where(cdf.loc['axl'])[0]]
    for pob in post_blocks:
        for prb in pre_blocks:
            stot = int(np.sum(W[np.ix_(pob,prb)]))
            rolls = np.random.rand(stot)
            pre = np.random.choice(prb,stot)
            post = np.random.choice(pob,stot)
            for i in range(stot):
                if rolls[i] < fn:
                    false_negatives[post[i],pre[i]] += 1
    stot = np.zeros(W.shape[0])
    for i in range(W.shape[0]):
        add_rolls = np.random.rand(int(totalInputs[i] - np.sum(W[i,:])))
        sub_rolls = np.random.rand(int(totalInputs[i] - np.sum(W[i,:])))
        stot[i] = totalInputs[i] + np.sum(add_rolls < fn) - np.sum(sub_rolls < fp)
        for j in range(W.shape[0]):
            rolls = np.random.rand(int(W[i,j]))
            false_positives = np.sum(rolls < fp)
            W[i,j] = W[i,j] - false_positives
            stot[i] = stot[i] - false_positives
    W = W + false_negatives
    for i in range(W.shape[0]):
        stot[i] += np.sum(false_negatives[i,:])
    return W,stot
    
#connMatFile = 'data/ConnMatrixPre_cleaned.mat'
connMatFile = 'data/ConnMatrix_CO_top500_2blocks_gamma038_08062020.mat'
connMat = scipy.io.loadmat(connMatFile)
connMatDict = list(connMat)
connMat = np.float32(connMat[connMatDict[-1]])
N = np.shape(connMat)[0]
print(N)

totalInputFile = 'data/totalInputs_CO_top500_2blocks_gamma038_08062020.mat'
totalInputs = scipy.io.loadmat(totalInputFile)
totalInputsDict = list(totalInputs)
totalInputs = np.int32(totalInputs[totalInputsDict[-1]])
totalInputs = np.ravel(totalInputs)

# load cellIDs
cellIDFile  = 'data/cellIDType_CO_top500_2blocks_gamma038_08062020.mat'
cellIDs = scipy.io.loadmat(cellIDFile)
cellIDFileDict = list(cellIDs)
cellIDs = cellIDs[cellIDFileDict[-1]]
cellIDs_unique = set(cellIDs)

matOrderFile  =  'data/MatOrder_CO_top500_2blocks_gamma038_08062020.mat'
matOrder = scipy.io.loadmat(matOrderFile)
idpos = matOrder['MatOrder_CO_top500_2blocks_gamma038_08062020'][0]

louvain_blocks = []
blockFile = 'data/block1_cleaned_gamma038_08072020.mat'
block1 = scipy.io.loadmat(blockFile)
louvain_blocks.append(block1['block1_cleaned'][0][0][0][0])
blockFile = 'data/block2_cleaned_gamma038_08072020.mat'
block2 = scipy.io.loadmat(blockFile)
louvain_blocks.append(block2['block2_cleaned'][0][0][0][0])

lbid = generate_blockIds(idpos,louvain_blocks)
sbmid = copy.deepcopy(lbid)
    
# get location of neurons
cellLocations =  np.array([(cellIDs == '_Int_'),(cellIDs == 'Ibn_m'),(cellIDs == 'Ibn_i'),(cellIDs == '_MOs_'),(cellIDs == '_Axlm'), (cellIDs == '_Axl_'), (cellIDs == '_DOs_'),(cellIDs == 'ABD_m'),(cellIDs == 'ABD_i'), (cellIDs == 'vSPNs')])
cellNames = ('integ','Ibnm','Ibni','MO','axlm','axl','vest','abdm','abdi','vspns')
lb_cdf = pandas.DataFrame(cellLocations,cellNames)

connMat[:,lb_cdf.loc['abdm']] = 0
connMat[:,lb_cdf.loc['abdi']] = 0

jConnMat, jTotIn = jitter_connections(connMat,lb_cdf,totalInputs)
jlbid = clusterCenterLouvain(0.38,customMat=jConnMat)
if np.sum(cellIDs[jlbid == 0] == '_Axl_') < np.sum(cellIDs[jlbid == 0] == '_Int_'):
    jlbid[jlbid == 0] = 99
    jlbid[jlbid == 1] = 0
    jlbid[jlbid == 99] = 1
cellLocations =  np.array([(lbid == 1) & (cellIDs != '_DOs_'),(cellIDs == 'Ibn_m'),(cellIDs == 'Ibn_i'),(cellIDs == '_MOs_'),(cellIDs == '_Axlm'), (lbid == 0), (cellIDs == '_DOs_'),(cellIDs == 'ABD_m'),(cellIDs == 'ABD_i'), (cellIDs == 'vSPNs')])
jcdf = pandas.DataFrame(cellLocations,cellNames)
jWnorm = np.zeros(connMat.shape)
lb_Wnorm = np.zeros(connMat.shape)
for i in np.arange(connMat.shape[0]):
    if jTotIn[i]>0:
        jWnorm[i,:] = jConnMat[i,:] / jTotIn[i,None]
    if totalInputs[i]>0:
        lb_Wnorm[i,:] = connMat[i,:] / totalInputs[i,None]
jWnorm = final_adjustments(jWnorm,jcdf)
y,v = sorted_eigs(jWnorm)
print(y[0])
lb_Wnorm = final_adjustments(lb_Wnorm,lb_cdf)
y,v = sorted_eigs(jWnorm)
print(y[0])
slopes = simulate_ks(jWnorm,np.real(y[0]),ynew=0.9,tau=1)
sf = 2.574 / np.mean(slopes[jcdf.loc['integ']])
jslopes = slopes*sf
sbmid2 = clusterModOSBM(2,'Louvain')
sbmid7 = clusterModOSBM(7,'Louvain')
y,v = sorted_eigs(jConnMat)
print(y[0])
y,v = sorted_eigs(connMat)
print(y[0])

data = {'connMat':jConnMat,
        'totalInputs':jTotIn,
        'slopes':jslopes,
        'eigenvalues':y,
        'modO':lbid,
        '2 Block SBM':sbmid2,
        '7 Block SBM':sbmid7}
f = open(f'data/jitter_{sys.argv[1]}.data','wb')
pickle.dump(data,f)
f.close()