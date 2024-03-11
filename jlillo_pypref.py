import matplotlib as mpl
# Use tex - a pain to setup, but it looks great when it works !
#mpl.rc('MacOSX')
# Default fonts - only used if usetex = False. The fontsize remains important though.
mpl.rc('font',**{'family':'sans-serif', 'serif':['Computer Modern Serif'],
             'sans-serif':['Tahoma'], 'size':18,
             'weight':500, 'variant':'normal'})
#mpl.rc('font',family='Helvetica')
# You can set many more things in rcParams, like the default label weight, etc ...
mpl.rc('axes',**{'labelweight':'normal', 'linewidth':1})
mpl.rc('ytick',**{'major.pad':12, 'color':'k'})
mpl.rc('xtick',**{'major.pad':8,})
mpl.rc('contour', **{'negative_linestyle':'solid'}) # dashed | solid
# The default matplotlib LaTeX - only matters if usetex=False.
mpl.rc('mathtext',**{'default':'regular','fontset':'cm',
                 'bf':'monospace:bold'})
# This is where the magic happens !
mpl.rc('text', **{'usetex':False})
# And this is how one can load exotic packages to fullfill one's dreams !
#mpl.rc('text.latex',preamble=r'\usepackage{cmbright},\usepackage{relsize},\usepackage{upgreek}, \usepackage{amsmath}')
