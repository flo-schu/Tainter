import matplotlib.pyplot as plt
import numpy as np
import os

def write_filename(fct, args, folder_name="", random_repeat = ""):
    """function to write a filename from the executed function and its arguments"""
    if not os.path.exists("./plots/"+folder_name):
        os.makedirs("./plots/"+folder_name)
    if not os.path.exists("./data/"+folder_name):
        os.makedirs("./data/"+folder_name)
    str1 = fct + "__" + "rand" + str(random_repeat) + '__' +'-'.join([str(args['network']) , str(args['N']) , str(args['k']) , str(args['p'])]) + "_"
    str2 = '-'.join([str(args['first_admin']) , str(args['choice'])]) + "_"
    str3 = '-'.join([str(args['a']) , str(args['stress']) , str(args['shock']), str(args['threshold']) , str(args['eff']), str(args['death_energy_level']), str(args['exploration']), str(args['tmax'])])
    return str1 + str2 + str3

def legend(args,lines = 2):
    toplines = int(round(len(args.keys())/lines,0))
    lastline = len(args.keys())%toplines

    params = ""
    for i in np.arange(lines):
        params = params + str(list(args.items())[i*toplines:(i+1)*toplines])
        if i < (lines-1):
            params = params + "\n"

    return params


def node_dist(history, args, filename, folder_name):
    fig = plt.figure(figsize = (12,10))
    plt.subplot(311)
    plt.plot(np.array(history['Administration']) / args['N'], label="Administration share")
    plt.plot(np.array(history['Labourers']) / args['N'], label="Labourer share")
    plt.plot(np.array(history['coordinated Labourers']) / args['N'], label="coord. Labourer share")
    plt.xlabel("time")
    plt.ylabel("node distribution")
    plt.legend()
    params = str(list(args.items())[0:7])+"\n"+str(list(args.items())[7:])
    plt.annotate(params, (0,0), (0, -40), xycoords='axes fraction', textcoords='offset points', va='top', color = "grey")
    plt.savefig('./plots/' + folder_name + '/nodedistribution__' + filename +  '.png', bbox_inches='tight')
    return fig


def convert_to_array(my_object = None, my_dict = None):
    if isinstance(my_object, np.ndarray):
        return my_object
    if my_object != None:
        if isinstance(my_object, str):
            assert isinstance(my_dict, dict)
            try:
                return np.array(my_dict[my_object])
            except KeyError:
                print("variable not in dictionary")
                pass
        elif isinstance(my_object, list):
            np.array(my_object)
        else:
            print('Something went wrong.')
    else:
        pass

def generic(args, filename, plot_filename,folder_name="", my_dict = None, hline = False, xaxis = None, yaxis1 = None, yaxis2 = None, yaxis3 = None, xlab = None, ylab1 = None, ylab2 = None, ylab3 = None, normalize = [0,0,0], lines = 2):
    x = convert_to_array(xaxis, my_dict)
    y1 = convert_to_array(yaxis1,my_dict)
    y2 = convert_to_array(yaxis2,my_dict)
    y3 = convert_to_array(yaxis3,my_dict)

    if normalize[0] == 1: y1 = y1 / args['N']
    if normalize[1] == 1: y2 = y2 / args['N']
    if normalize[2] == 1: y3 = y3 / args['N']
    if ylab1 == None: ylab1 = yaxis1
    if ylab2 == None: ylab2 = yaxis2
    if ylab3 == None: ylab3 = yaxis3
    if xlab == None: xlab = xaxis

    fig = plt.figure(figsize = (12,10))
    plt.subplot(311)
    if xaxis == None:
        if yaxis1 != None: plt.plot(y1, label =  ylab1)
        if yaxis2 != None: plt.plot(y2, label =  ylab2)
        if yaxis3 != None: plt.plot(y3, label =  ylab3)
    else:
        if yaxis1 != None: plt.plot(x, y1, label =  ylab1)
        if yaxis2 != None: plt.plot(x, y2, label =  ylab2)
        if yaxis3 != None: plt.plot(x, y3, label =  ylab3)

    try:
        plt.axhline(y=hline, color = "grey")
    except TypeError:
        pass

    plt.xlabel(xlab)
    plt.legend()
    params = legend(args,lines)
    #params = str(list(args.items())[0:7])+"\n"+str(list(args.items())[7:])
    plt.annotate(params, (0,0), (0, -40), xycoords='axes fraction', textcoords='offset points', va='top', color = "grey")
    plt.savefig('./plots'+ folder_name + '/' + plot_filename + '__' + filename +  '.png', bbox_inches='tight')

def extract_history(entry, my_list, index = -1, ie1 = 0, ie2 = 0 ):
    record = list()

    if isinstance([my_list[0][entry]][index], list):
        for i in my_list:
            record.append([i[entry]][index][ie1][ie2])
        return record
    else:
        for i in my_list:
            record.append([i[entry]][index])
        return record

def xyplot(x,y,title,args,filename,folder_name="", xlab = None, ylab = None):
    plt.figure(figsize = (12,10))
    plt.subplot(311)
    plt.plot( x, y)
    if xlab != None: plt.xlabel(xlab)
    if ylab != None: plt.ylabel(ylab)
    plt.title(title)
    params = str(list(args.items())[0:7])+"\n"+str(list(args.items())[7:])
    plt.annotate(params, (0,0), (0, -40), xycoords='axes fraction', textcoords='offset points', va='top', color = "grey")
    #plt.legend()
    plt.savefig('./plots' + folder_name + '/' + title + '__' + filename + '.png', bbox_inches = 'tight')
    plt.show()

def twoDplot(xarray, yarray, colvar, ncol, nrow, title, args, folder_name, filename, xlab = None, ylab = None, zlab = None, lines = 2):
    grid = colvar.reshape((ncol, nrow))
    grid = np.flipud(grid.T)

    plt.imshow(grid, extent=(xarray.min(), xarray.max(), yarray.min(), yarray.max()),
               interpolation='nearest')
    plt.imshow(grid, aspect = 'auto', extent=(xarray.min(), xarray.max(), yarray.min(), yarray.max()), interpolation='nearest')
    cbar = plt.colorbar()
    cbar.set_label(zlab, rotation=270)
    if xlab != None: plt.xlabel(xlab)
    if ylab != None: plt.ylabel(ylab)
    plt.title(title)
    params = legend(args, lines)
    plt.annotate(params, (0,0), (0, -40), xycoords='axes fraction', textcoords='offset points', va='top', color = "grey")
    plt.savefig('./plots/' + folder_name + '/' + title + '__' + filename + '.png', bbox_inches = 'tight')
    plt.show()


def scattercolor(x,y,colvar,args,title,filename,folder_name,xlab = None, ylab = None,marker = 'o'):
    #plt.figure(figsize = (12,11))
    #plt.subplot(311)
    cmap= plt.scatter(x, y, c=colvar, marker='o')

def scattercolor(x,y,colvar,args,title,filename,folder_name,xlab = None, ylab = None,marker = 'o', transparency = None):
    plt.figure(figsize = (12,11))
    plt.subplot(311)
    cmap= plt.scatter(x, y, c=colvar, marker='o',alpha = transparency)

    plt.colorbar(cmap)
    if xlab != None: plt.xlabel(xlab)
    if ylab != None: plt.ylabel(ylab)
    plt.title(title)

    params = str(list(args.items())[0:7])+"\n"+str(list(args.items())[7:])
    plt.annotate(params, (0,0), (0, -40), xycoords='axes fraction', textcoords='offset points', va='top', color = "grey")
    # plt.legend()
    plt.savefig('./plots/' + folder_name + '/' + title + '__' + filename + '.png', bbox_inches = 'tight')
    plt.show()
