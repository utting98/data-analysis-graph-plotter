"""
General Graph Plotter
Joshua Utting
23/11/2019

This code acts as an extended version of the LSFR.py file distributed by the UoM physics department. This code can fit any general
polynomial to comma seperated data from a .txt or .csv file formatted as x,y,y_err. It will give results for all of the fitting
coefficients for a positive integer polynomial with corresponding uncertainties on each parameter as well as a reduced chi squared
value for the fit. If an x-ordinate is specified for a non-linear polynomial, the code will also specify the gradient at that
x-ordinate with an uncertainty. The other fitting option is called custom law, here you can specify an equation you wish to try and
fit to the data with some fitting parameters. Fitting parameters are constants that the code is allowed to adjust to make a better
fit of the data, to run this method at least one fitting parameter is required so that it can optimise around something. For example
if the function you are fitting has an angle in it and you take a cosine you can replace cos(x) with cos(x+phi) where phi is an
arbitrary phase shift, if you are not expecting a phase shift in your data this value should be approximately 0. Fitting parameter
values will also have an uncertainty on them. Initial gueses of fitting parameter values must be provided as somewhere to start
optimising around, these do not need to be exact just order of magnitude estimates and the code will manage to get to the right
value. If the estimate is too far away this will be obvious from the graph as the fitted line will be too far from the data so
you can determine whether the estimate needs to be higher or lower based on the position. So the final outputs of custom mode
will be the fitting parameters with uncertainties and the reduced chi squared value.

"""

#imports list
import numpy as np
import matplotlib.pyplot as plt
from math import *
from scipy import optimize
from IPython import get_ipython
from tkinter import *
import webbrowser

get_ipython().run_line_magic('matplotlib', 'qt') #set graphs to display in window instead of console

#fitting function for custom mode, takes argument of the x values and unpacks a list of arguments that are the fitting parameters
def fitting(xvals,*arglist):
    global equation, userfittingparams, firstrun, oldparams #access required globals
    if(firstrun==True): #if this is the first attempt at fitting
        #loop over and replace all of the fitting parameters with their guess values 
        for i in range(0,len(userfittingparams)):
            equation = equation.replace(str(userfittingparams[i]),str(arglist[i]))
        oldparams = arglist #set the old version of the parameters to their current version
        firstrun = False #mark the first run is completed
    else: #if not the first run
        #loop over the old fitting parameter values and replacec with the new estimate by the optimising function
        for i in range(0,len(userfittingparams)):
            equation = equation.replace(str(oldparams[i]),str(arglist[i]))
        oldparams = arglist #update list of the old parameters
    #define empty list of y valyes for the updated equation
    vallist = []
    for i in range(0,len(xvals)): #loop over the x values
        newequation = equation.replace('x',str(xvals[i])) #replace references to x with the current x value fromthe list
        if(str('e'+str(xvals[i])+'p') in newequation): #a fix to prevent replacing the x in exp command being raplced by a number 
            newequation = newequation.replace(str('e'+str(xvals[i])+'p'),'exp')
        else:
            pass
        val = eval(newequation) #calculate the y value of this point
        vallist.append(val) #append the y value to the list
    return vallist #return the y values to the optimisation function for comparison
    
#function to fit and plot a polynomial to the data
def powerplot(xvals, yvals, error, power, xtitle, ytitle, plottitle, outfilepath, gradpoints):
    #this has already been validated to work so float the power
    power  = int(power)
    
    #plot a new figure 1
    fig1 = plt.figure(1,figsize=(9,6))
    plt.subplot(211) #add a subplot for fit
    #plot the real data values with error bars and no line with set marker size and colours
    plt.errorbar(xvals, yvals, yerr = error, ecolor = 'red', linestyle = 'None', marker = 'o', markersize = 3)
    plt.grid(True) #add a grid
    plt.xlabel(xtitle) #add an x label to the graph from the user input
    plt.ylabel(ytitle) # add a y label to the graph from the user input
    plt.title(plottitle) #add a title to the graph from the user input
    plt.tight_layout() #stop labels from overlapping
    
    #calculate a polyfit of the x and y values to whatever order the user has specified and return the parameters and covariance matrix
    p, cov = np.polyfit(xvals, yvals, power, cov = True)
    fitp = np.polyval(p, xvals) #calculate new y values from the x values and the fitting parameters
    plt.plot(xvals, fitp) #plot the calulated fitting line over the data
    
    residuals = (yvals - fitp) #work out a list of residuals
    chi2 = 0 #set initial chi squared as 0
    chi2data = np.empty(len(yvals), dtype=float) #define list for contribution of each point to be stored for more thorough analysis if user wants to investigate through the code
    for a in range(0, len(yvals)): #loop over the y values
        chi2data[a] = ((yvals[a] - fitp[a])/error[a])**2 #calculate chi squared value for that point
        chi2 += chi2data[a] #add that chi squared value to the existing total
    N = (len(yvals)-len(p)) #work out degrees of freedom as number of points subtract the number of fitting parameters
    redchi2 = chi2/N #calculate a reduced chi squared as chi squared over the degrees of freedom
    
    plt.subplot(212) #add a subplot for residuals
    y0 = [0]*len(yvals) #create an array of zeroes to draw a line at y=0 across the residual graph
    #plot the residual values with no line and coloured errorbars with distinct marker
    plt.errorbar(xvals, residuals, yerr = error, ecolor = 'red', linestyle = 'None', marker = 'o', markersize = 3)
    plt.grid(True) #plot a grid
    plt.plot(xvals,y0) #plot a line through y=0
    plt.title('Residuals') #add a title of residuals
    cov2 = cov*(len(yvals)-len(p)-2)/chi2 #calculate a truer covariance matrix of the fitting parameters
    fittingerror = np.sqrt(np.diag(cov2)) #get a list of the uncertainties on the fitting parameters
    #check if a gradient ordinate was specified
    if(gradpoints!='' and power!=1):
        coordinates = float(gradpoints) #this has already been validated so float it
        powercoeff = power #store a duplicated of the power to iterate over
        gradeqn='' #define the gradient equation as an empty string
        for i in range(0,powercoeff): #loop over the number of terms to calculate the gradient
            if(powercoeff!=0): #if it is not the intercept value
                derivativecoeff = float(p[i])*powercoeff #the derivative coefficient is it's original coefficient times its power
                powercoeff-=1 #the power on the x value is reduced by one
                if(powercoeff==0): #if it's intercept of the gradient
                    string = '(%s)' % (derivativecoeff) #make string of new coefficient, no x term is needed
                else: #if it's not the intercept
                    string = '(%s*x**%s)' % (derivativecoeff,powercoeff) #add the new coefficient times x to its reduced power to a new string
                if(powercoeff==power-1): #if this is the first iteration
                    gradeqn = string #make gradient equation the new string
                else: #if not the first iteration
                    gradeqn = gradeqn+'+'+string #add the new string to the existing gradient equation
            else:
                pass

        gradeqnsubbed = gradeqn.replace('x',str(coordinates)) #when full derivative equation found replace x with the x ordinate the gradient is wanted for
        gradval = eval(gradeqnsubbed) #evaluate the gradient string to get a value

        gradienterror = np.sqrt(np.sum(fittingerror[:-1]**2)) #work out the  uncertainty on the gradient value
    
    else: #if gradient not wanted/needed
        pass #do nothing
    
    powercoeff = power #reset powercoeff for iterating
    #add label informing of fitting parameters to scrollable frame
    resultslabel = Label(subframe,text='Fitting Parameters:',relief='solid').grid(row=0,column=0,sticky='nsew') 
    #add label of the reduced chi squared value to the scrollable window
    chilabel = Label(subframe, text=('Reduced Chi Squared: %8.6f' % redchi2),relief='solid').grid(row=1,column=0,sticky='nsew')
    labelrow = 2 #set variable to define which row new parameters need to be displayed on
    if(powercoeff==1): #if the graph is linear there will only be a gradient and y intercept so display those with uncertainty
        gradlabel = Label(subframe,text=('m: %.5e ± %.5e' % (p[0],fittingerror[0])),relief='solid').grid(row=2,column=0,sticky='nsew')
        clabel = Label(subframe,text=('c: %.5e ± %.5e' % (p[1],fittingerror[1])),relief='solid').grid(row=3,column=0,sticky='nsew')
    else: #if it is higher order than linear
        for i in range(0,len(p)): #loop over the number of coefficients
            if(powercoeff!=0): #special case conditions for tidier display, if not down to 0th order ie y intercept
                if(powercoeff==1): #if it is x^1 just display as x with its coefficients and uncertainty
                    Label(subframe,text=('x Coefficient: %.5e ± %.5e' % (p[i],fittingerror[i])),relief='solid').grid(row=labelrow,column=0,sticky='nsew')
                else: #if it is not the x coefficient diplay it as x^powercoeff with its coefficient and uncertainty
                    Label(subframe,text=('x^%s Coefficient: %.5e ± %.5e' % (powercoeff,p[i],fittingerror[i])),relief='solid').grid(row=labelrow,column=0,sticky='nsew')
            else: #if it is x^0 then it is the y intercept so display it as just a coefficient with uncertainty
                Label(subframe,text=('y Intercept: %.5e ± %.5e' % (p[i],fittingerror[i])),relief='solid').grid(row=labelrow,column=0,sticky='nsew')
            powercoeff-=1 #remove one from the powercoeff as when powercoeff=0 the whole equation is iterated over
            labelrow+=1 #add one to the label row so the new label displays on the next
    if(gradpoints!=''): #if they specified a gradient ordinate display the gradient with uncertainty at that point
        Label(subframe,text=('Gradient At x=%s: %.5e ± %.5e' % (coordinates,gradval,gradienterror)),relief='solid').grid(row=labelrow,column=0,sticky='nsew')
    else:
        pass
    plt.tight_layout() #ensure graph labels do not overlap
    plt.show() #display the graphs
    plottitle = plottitle.replace(' ','') #remove spaces from the title string for naming the file
    try: #try and save the file to the output path specified
        if(outpath!=''): #if a path was specified save it to that with as the plottitle.png
            titlestring = outpath+'/'+plottitle+'.png'
        else: #if no path was specified save it to the current directory of the code as plottitle.png
            titlestring = './' + plottitle+'.png'
        fig1.savefig(titlestring, bbox_inches="tight")
    except: #if the specified output path does not exist then warn the user and inform them it will save to the directory of the code
        errorwarning('Error:\nOutput directory not found, graph will be saved to the same directory as this code by default when this window is closed.\nCheck to see if you made a typo when specifying the output path.')
        titlestring = './' + plottitle+'.png'
        fig1.savefig(titlestring, bbox_inches="tight")

#callback function to open a link in the default web browser
def callback(url):
    webbrowser.open_new(url)

#function to configure a scroll region on a canvas
def scrollfunc2(event):
    global canvas2
    canvas2.configure(scrollregion=canvas2.bbox("all"),width=700,height=500)

#fuction to create a new canvvas inside a frame then put a new frame inside that canvas to allow it to be scrolled
def canvascreate2():
    global canvas2, functionsframe, subframe2
    #create a canvas in the frame that needs to be scrollable
    canvas2=Canvas(functionsframe)
    subframe2=Frame(canvas2) #create another frame inside that canvas
    myscrollbar=Scrollbar(functionsframe,orient="vertical",command=canvas2.yview) #define a scrollbar in the canvas host frame and bind it to a command to navigate the y direction of the canvas
    canvas2.configure(yscrollcommand=myscrollbar.set) #configure canvas2 for scrolling
    
    myscrollbar.pack(side="right",fill="y") #pack the scrollbar to the right of the frame and fill the frame vertically
    canvas2.pack(side="top") #pack the canvas in the host frame
    canvas2.create_window((200,0),window=subframe2,anchor='n') #create a window in the canvas that corresponds to subframe2 and anchor it to the top
    subframe2.bind("<Configure>",scrollfunc2) #bind the function scrollfunc2 to the frame to allow a scrollable region of subframe2 with size defined in scrollfunc2
    return

#function to provide a help popup window if user is unsure about how to use custom mode
def customhelp():
    global canvas2, functionsframe, subframe2
    helpwin = Tk() #define the window as helpwin
    helpwin.title('Help') #set tht title of the window
    helpwin.attributes("-topmost", True) #make the helpwindow appear on the top level so user sees it
    #label prociding some information about custom mode and some basics of how to use it
    infotext = Label(helpwin,text='Custom mode can be difficult to understand.\nYou can find a video explanation on how to use it here:').grid(row=0,column=0)
    #add text saying link to the video in blue and bind the callback function to it with a link to a video explaining how to use it
    link1 = Label(helpwin, text="Link To Video", fg="blue", cursor="hand2")
    link1.grid(row=1,column=0)
    link1.bind("<Button-1>", lambda e: callback("https://drive.google.com/open?id=1OIVKYUg6kBFeRrtZmX324GtWr3FjH6xE"))
    #define new frame in the helpwindow where function formatting will be listed
    functionsframe = Frame(helpwin)
    functionsframe.grid(row=2,column=0)
    #create a new scrollable canvas to display the list of functions in
    canvascreate2()
    #some extra information to watch out for when using custom mode
    constantslabel = Label(subframe2,text='Please replace constants with their values in the equations as the code will not recognise constant names.').grid(row=0,column=0,columnspan=4)
    naminglabel = Label(subframe2,text='Please also ensure that your fitting parameter names do not containt "x" as the code will try to replace part of the word with x values.\nNote all trigonometric functions work in radians\nAlso note functions may not be compatible with complex numbers.').grid(row=1,column=0,columnspan=4)
    
    #the following is code to define a grid of labels that show the name/formatting of a mathematical funciton on the left and how to format it in the code on the right
    Label(subframe2,text='Function Name',relief='solid').grid(row=2,column=1,sticky='nsew')
    Label(subframe2,text='How To Write In Program',relief='solid').grid(row=2,column=2,sticky='nsew')
    
    Label(subframe2,text='sin(x)',relief='solid').grid(row=3,column=1,sticky='nsew')
    Label(subframe2,text='sin(x)',relief='solid').grid(row=3,column=2,sticky='nsew')
    
    Label(subframe2,text='cos(x)',relief='solid').grid(row=4,column=1,sticky='nsew')
    Label(subframe2,text='cos(x)',relief='solid').grid(row=4,column=2,sticky='nsew')
    
    Label(subframe2,text='tan(x)',relief='solid').grid(row=5,column=1,sticky='nsew')
    Label(subframe2,text='tan(x)',relief='solid').grid(row=5,column=2,sticky='nsew')
    
    Label(subframe2,text='arcsin(x)',relief='solid').grid(row=6,column=1,sticky='nsew')
    Label(subframe2,text='asin(x)',relief='solid').grid(row=6,column=2,sticky='nsew')
    
    Label(subframe2,text='arccos(x)',relief='solid').grid(row=7,column=1,sticky='nsew')
    Label(subframe2,text='acos(x)',relief='solid').grid(row=7,column=2,sticky='nsew')
    
    Label(subframe2,text='arctan(x)',relief='solid').grid(row=8,column=1,sticky='nsew')
    Label(subframe2,text='atan(x)',relief='solid').grid(row=8,column=2,sticky='nsew')
    
    Label(subframe2,text='sin^2(x)',relief='solid').grid(row=9,column=1,sticky='nsew')
    Label(subframe2,text='sin(x)**2',relief='solid').grid(row=9,column=2,sticky='nsew')
    
    Label(subframe2,text='cos^2(x)',relief='solid').grid(row=10,column=1,sticky='nsew')
    Label(subframe2,text='cos(x)**2',relief='solid').grid(row=10,column=2,sticky='nsew')
    
    Label(subframe2,text='tan^2(x)',relief='solid').grid(row=11,column=1,sticky='nsew')
    Label(subframe2,text='tan(x)**2',relief='solid').grid(row=11,column=2,sticky='nsew')
    
    Label(subframe2,text='sinh(x)',relief='solid').grid(row=12,column=1,sticky='nsew')
    Label(subframe2,text='sinh(x)',relief='solid').grid(row=12,column=2,sticky='nsew')
    
    Label(subframe2,text='cosh(x)',relief='solid').grid(row=13,column=1,sticky='nsew')
    Label(subframe2,text='cosh(x)',relief='solid').grid(row=13,column=2,sticky='nsew')
    
    Label(subframe2,text='tanh(x)',relief='solid').grid(row=14,column=1,sticky='nsew')
    Label(subframe2,text='tanh(x)',relief='solid').grid(row=14,column=2,sticky='nsew')
    
    Label(subframe2,text='arcsinh(x)',relief='solid').grid(row=15,column=1,sticky='nsew')
    Label(subframe2,text='asinh(x)',relief='solid').grid(row=15,column=2,sticky='nsew')
    
    Label(subframe2,text='arccosh(x)',relief='solid').grid(row=16,column=1,sticky='nsew')
    Label(subframe2,text='acosh(x)',relief='solid').grid(row=16,column=2,sticky='nsew')
    
    Label(subframe2,text='arctanh(x)',relief='solid').grid(row=17,column=1,sticky='nsew')
    Label(subframe2,text='atanh(x)',relief='solid').grid(row=17,column=2,sticky='nsew')
    
    Label(subframe2,text='e^x',relief='solid').grid(row=18,column=1,sticky='nsew')
    Label(subframe2,text='exp(x)',relief='solid').grid(row=18,column=2,sticky='nsew')
    
    Label(subframe2,text='ln(x)',relief='solid').grid(row=19,column=1,sticky='nsew')
    Label(subframe2,text='log(x)',relief='solid').grid(row=19,column=2,sticky='nsew')
    
    Label(subframe2,text='log10(x)',relief='solid').grid(row=20,column=1,sticky='nsew')
    Label(subframe2,text='log10(x)',relief='solid').grid(row=20,column=2,sticky='nsew')
    
    Label(subframe2,text='loga(x)',relief='solid').grid(row=21,column=1,sticky='nsew')
    Label(subframe2,text='log(x,a)',relief='solid').grid(row=21,column=2,sticky='nsew')
    
    Label(subframe2,text='x^a',relief='solid').grid(row=22,column=1,sticky='nsew')
    Label(subframe2,text='x**a',relief='solid').grid(row=22,column=2,sticky='nsew')
    
    Label(subframe2,text='[a-th or square root](x)',relief='solid').grid(row=23,column=1,rowspan=2,sticky='nsew')
    Label(subframe2,text='x**(1/a)',relief='solid').grid(row=23,column=2,sticky='nsew')
    Label(subframe2,text='sqrt(x)',relief='solid').grid(row=24,column=2,sticky='nsew')
    
    Label(subframe2,text='ax',relief='solid').grid(row=25,column=1,sticky='nsew')
    Label(subframe2,text='a*x',relief='solid').grid(row=25,column=2,sticky='nsew')
    
    Label(subframe2,text='a/x',relief='solid').grid(row=26,column=1,sticky='nsew')
    Label(subframe2,text='a/x',relief='solid').grid(row=26,column=2,sticky='nsew')
    
    Label(subframe2,text='a+x',relief='solid').grid(row=27,column=1,sticky='nsew')
    Label(subframe2,text='a+x',relief='solid').grid(row=27,column=2,sticky='nsew')
    
    Label(subframe2,text='a-x',relief='solid').grid(row=28,column=1,sticky='nsew')
    Label(subframe2,text='a-x',relief='solid').grid(row=28,column=2,sticky='nsew')
    
    helpwin.mainloop() #display and loop the help window

#define function to display error warnings where an error message is passed and displayed
def errorwarning(message):
    errorwin = Tk() #define error window
    errorwin.title('Error') #give window title of error
    warninglabel= Label(errorwin,text=message).grid(row=0,column=0) #text to display the passed error message
    okbutton = Button(errorwin,text='Ok',command=lambda: errorwin.destroy()) #ok button that closes the error window
    okbutton.grid(row=1,column=0)
    errorwin.mainloop() #display and loop the error window

#defining plotting function where code gets the user inputs and figures out what plot to do
def plot(dirpath,method,power,gradords,eqn,params,guess,xlabel,ylabel,graphtitle,outfilepath):
    global equation, userfittingparams, firstrun, subframe, root, paramsframe
    #if there are already fitting parameters listed remoce the frame of them to prevent overlap in display
    if(root.grid_slaves(row=10,column=1)!=[]):
        paramsframe.grid_forget()
    else:
        pass
    #create new scrollable frame for the fitting parameters incase there is a lot
    canvascreate()
    filepath = dirpath.get() #get input of path to file
    filepath = filepath.replace("\\","/") #replace \ with / as \ is a special character in python that starts indicating special formatting
    xtitle = xlabel.get() #get the x label text
    ytitle = ylabel.get() #get the y label text
    title = graphtitle.get() #get the title text
    version = method.get() #get either a vallue of one or two for polynomial and custom fit respectively
    outpath = outfilepath.get() #get output file directory
    outpath = outpath.replace("\\","/") #same as for input file replacements

    if(version==1): #only get the following if it is polynomial fit
        maxpower = power.get() #get highest power of x from user input
        coords = gradords.get() #get gradient x ordinate from input
    else: #only get the following if it is custom fit
        equation = eqn.get() #get equation string from user input
        fittingparameterstemp = params.get() #get the fitting parameters string from user input
        temp = guess.get() #get the guess values from user input
        paramguesses = [] #define empty list for floated version of parameter guesses
        templist = temp.split(',') #split the guesses on commas to make a list
        userfittingparams = fittingparameterstemp.split(',') #split the user fitting parameters on commas to make a list
        try: #try looping over the list of guesses and floating them, if successfuly append to the paramguesses list
            for i in range(0,len(templist)):
                paramguesses.append(float(templist[i]))
        except: #if a value cant be floated give an error message explaining the problem to the user then return without plotting
            errorwarning('Error:\nEither no parameter guesses were provided or a text string was provided instead of a number.\nPlease only enter comma separated numbers, one for each fitting parameter specified.')
            return
    
    if(title==''): #if not title was specified give an error message explaining the problem to the user then return without plotting
        errorwarning('Error:\nPleas specify a graph title, this will appear above your graph and is what the output file will be named as in your documents.')
        return
    else:
        pass
    
    #define empty list of x, y and error values
    x = [] 
    y = []
    err = []
    
    if(version==1): #if it is a polynomial fit
        
        try: #check that max power is a number, if not give an error message explaining the problem to the user then return without plotting
            float(maxpower)
        except:
            errorwarning('Error:\nPlease specify a positive integer power as the highest power of the polynomial you are trying to fit.\nFor example for a quartic fit you should type 4 and it will fit for parameters x^4 down to the y intercept.\nThis method only works for positive integer powers of x.\nIf you want non integer or negative powers please specify the whole equation in the Custom Equation section instead.')
            return
        
        if(float(maxpower)<=0 or not float(maxpower).is_integer()): #check if max power positive and an integer, if not give an error message explaining the problem to the user then return without plotting
            errorwarning('Error:\nPlease specify a positive integer power as the highest power of the polynomial you are trying to fit.\nFor example for a quartic fit you should type 4 and it will fit for parameters x^4 down to the y intercept.\nThis method only works for positive integer powers of x.\nIf you want non integer or negative powers please specify the whole equation in the Custom Equation section instead.')
            return
        else:
            pass
        
        if(coords!=''): #check if an input was specified for the x ordinate value
            try: #check if the value of the x ordinate is a float, if not give an error message explaining the problem to the user then return without plotting
                float(coords)
            except:
                errorwarning('Error:\nThe gradient X ordinate you specified was not a number.\nPlease either enter a number for the gradient at that X value or leave it blank to not get the gradient.')
                return
        else:
            pass
        
        try: #check if the data file can be found and opened to read, if not give an error message explaining the problem to the user then return without plotting
            file=open(filepath,'r')
        except:
            errorwarning('Error:\nFile cannot be found\nEnsure that you have typed the directory and file name correctly')
            return
        
        for line in file: #loop over each line in the file
            try: #try and split up each line on commas and append the floated values to the lists, if not give an error message explaining the problem to the user then return without plotting
                if(line!='\n'): #if the line isn't an acccidental blank new line
                    contents = line.split(',')
                    x.append(float(contents[0]))
                    y.append(float(contents[1]))
                    contents[2] = contents[2].strip('\n')
                    err.append(float(contents[2]))        
                else:
                    pass
            except:
                errorwarning('Error:\nAt least one value in your file is unable to be used.\nEnsure that all your values are numbers with no extra spaces or characters and there are no column headings in text form.\nAlso ensure values are comma seperated (file should be either .txt split by commas or .csv).')
                return
        
        file.close() #close the file
        powerplot(x,y,err,maxpower,xtitle,ytitle,title,outpath,coords) #run the plotting function for the polynomial fitting passing all relevant data
        
    else: #if the plot method is custom law instead
        
        if(equation==''): #check if equation input is blank, if it is give an error message explaining the problem to the user then return without plotting
            errorwarning('Error:\nAs you have selected custom mode please enter an equation to try and fit to the data.\n If you are unsure how to do this please select the Custom Help button.\nWithin that window there is a video tutorial and grid of functions you can use in combination with each other.')
            return
        else:
            pass
        
        #check if user fitting parameters are blank or if any of them are not in the equation, if so give an error message explaining the problem to the user then return without plotting
        if(userfittingparams==[] or userfittingparams=='' or userfittingparams==[''] or any(i not in equation for i in userfittingparams)):
            errorwarning('Error:\nPlease specify at least one fitting parameter used within you equation.\nA fitting parameter is a constant that the equation can adjust the value of to improve the fit of the data.\nFor example if you are fitting a trigonometric function to an x value that is an angle,\nyou can replace x with (x+phi) which is an arbitrary phase shift.\nIf you are not expecting a phase shift in your result the value of phi returned should be close to zero.\nThis is a further way of verifying that the fit is good as well as the reduced chi squared.\nFor more information please review the information by clicking the Custom Help button.\nIf you have provided at least one fitting parameter and you are seeing this message,\nthis means that at least one fitting parameter you listed is not used in the equation.')
            return
        else:
            pass
        
        #check if the parameter guesses are blank, if so give an error message explaining the problem to the user then return without plotting
        if(paramguesses==[] or paramguesses=='' or paramguesses==['']):
            errorwarning('Error:\nPlease enter some guess values for your fitting paramters.\nIf you do not understand what this means please review the video in the Custom Help section of this code.\nA list of guesses just have to be a similar order of magnitude and the code will optimise the values, they do not need to be exact.\nIf you cannot think of a reasonable value you can try guessing 0,0,0... for the number of fitting parameters you have specified.\nYou will be able to see by eye if these values are too large or too small based on the fit produced and you can adjust them accordingly.')
            return
        else:
            pass
        
        #check if the number of parameter guesses and the nubmer of fitting parameters are equal, if not give an error message explaining the problem to the user then return without plotting
        if(len(paramguesses)!=len(userfittingparams)):
            errorwarning('Error:\nPlease provide one guess for each fitting parameter comma separated in the order the parameters are listed.\nThis error means that the number of fitting parameters and number of guesses is not equal.\nDid you forget to provide a guess for one?')
            return
        else:
            pass
        
        #some catches to try and fix common formatting errors
        try:
            equation = equation.replace('^','**')
        except:
            pass
        try:
            equation = equation.replace('arcsin','asin')
        except:
            pass
        try:
            equation = equation.replace('arccos','acos')
        except:
            pass
        try:
            equation = equation.replace('arctan','atan')
        except:
            pass

        firstrun = True #mark it as the first run
    
        try: #check if the data file can be found and opened to read, if not give an error message explaining the problem to the user then return without plotting
            file=open(filepath,'r')
        except:
            errorwarning('Error:\nFile cannot be found\nEnsure that you have typed the directory and file name correctly')
            return
        
        for line in file: #loop over the lines in the file
            try: #try and split up each line on commas and append the floated values to the lists, if not give an error message explaining the problem to the user then return without plotting
                if(line!='\n'): #check that the line isn't an accidental blank line
                    contents = line.split(',')
                    x.append(float(contents[0]))
                    y.append(float(contents[1]))
                    contents[2] = contents[2].strip('\n')
                    err.append(float(contents[2]))        
                else:
                    pass
            except:
                errorwarning('Error:\nAt least one value in your file is unable to be used.\nEnsure that all your values are numbers with no extra spaces or characters and there are no column headings in text form.\nAlso ensure values are ccomma seperated (file should be either .txt split by commas or .csv.')
                return
        
        file.close() #close the file
        
        #optimise the fit where the new y values are calculated from the fitting function, the x and y values are read from the file and the parameter guesses are specified from the user input
        try:
            fittingparams = optimize.curve_fit(fitting,x,y,p0=paramguesses) 
        except:
            errorwarning('Error:\nYour equation could not be evaluated. If you are seeing this message ensure none of your fitting parameters contain the letter x\n This is because this can cause parts of their names to be replaced with numbers triggering a formatting error.\nIf this is not the case ensure that your equation is formatted correctly, click Custom Help button for details.')
            return
        ans,cov = fittingparams #store the results of the optimised version as answer and covariance
        
        paramvals = [0]*len(userfittingparams) #create list of parameter values
        paramerrs = [0]*len(userfittingparams) #create list of parameter errors
        for i in range(0,len(paramvals)): #loop over the parameter values and store them as the corresponding answer values
            paramvals[i] = ans[i]
        
        fitvals = fitting(x,*paramvals) #get the fitting y values 
        chisquared = 0 #Define the chi squared value for this fit
        chisquareddata = np.empty(len(y), dtype=float) #Define an empty array to fill
        for a in range(0, len(y)):
            chisquareddata[a] = ((y[a] - fitvals[a])/err[a])**2 #Calulate the contribution to the chi squared value of each point for Cox-Voinov
            chisquared += chisquareddata[a] #Calculate the value of chi squared as the sum of the previous values
        N = (len(y)-len(ans)) #Define degrees of freedom as number of fitting parameters subtracted from the number of points
        reducedchisquared = chisquared/N #Calculate reduced chi squared as chi squared over degrees of freedom
        cov2 = cov*(len(y)-len(userfittingparams)-2)/chisquared #calculate a new covariance matrix for the fitting paramaeter errors
        
        residuals = [] #define empty residuals list
        for i in range(0,len(y)): #loop over the number of y values
            residuals.append(y[i]-fitting(x,*paramvals)[i]) #append the residuals with the difference between the y values and the fitting values
        
        for i in range(0,len(paramvals)): #loop over the parameter values size
            paramerrs[i] = np.sqrt(np.diag(cov2))[i] #calculate errors on fitting parameters as the square root of the diagonal components of the covariance matrix
        
        fig1 = plt.figure(1,figsize=(9,6)) #plot new figure
        plt.subplot(211) #add new subplot to figure
        plt.errorbar(x,fitting(x,*paramvals)) #plot the fitting line calculated
        plt.errorbar(x,y,yerr=err,linestyle = 'None') #plot the values with their uncertainties
        plt.title(title) #Add the title to the plot from specified user input
        plt.xlabel(xtitle) #Add the label to the x axis from specified user input
        plt.ylabel(ytitle) #Add the label to the y axis from specified user input
        plt.grid() #add grid to the plot
        
        plt.subplot(212) #add new subplot to the figure
        r=[0]*len(y) #create list of 0 values for plotting y=0 graph
        plt.errorbar(x,residuals,yerr = err, linestyle = 'None') #Plot calculated residuals data with no line
        plt.plot(x,r) #Plot y=0 line to see distance to residuals
        plt.grid() #Plot a grid on the axes
        plt.title("Residuals") #Add the title to the plot
        plt.tight_layout() #stop labels overlapping
        plt.show() #show the plot
        
        #display label stating that fitting parameters will follow
        resultslabel = Label(subframe,text='Fitting Parameters:',relief='solid').grid(row=0,column=0,sticky='nsew')
        #display label showing reduced chi squared value for fit
        chilabel = Label(subframe, text=('Reduced Chi Squared: %8.6f' % reducedchisquared),relief='solid').grid(row=1,column=0,sticky='nsew')
        labelrow = 2 #variable to determine what row the next parameter should be displayed on
        for i in range(0,len(paramvals)): #loop over the number of parameter values
            #define the label text as the [fitting parameter]: value ± uncertainty
            textstring = (userfittingparams[i]+': %.5e ± %.5e' % (paramvals[i],paramerrs[i])) 
            #create and display label of that fitting parameter text
            Label(subframe,text=textstring,relief='solid').grid(row=labelrow,column=0,sticky='nsew')
            labelrow+=1 #increase the next row value by one
        
        plottitle = title.replace(' ','') #replace spaces in the title with nothing to make file title
        
        try: #if the outfile path has been specified try and save the figure to it
            if(outpath!=''):
                titlestring = outpath+'/'+plottitle+'.png'
            else:
                titlestring = './' + plottitle+'.png'
            fig1.savefig(titlestring, bbox_inches="tight")
        except: #if the outfile path has failed inform the user that it could not find it so the graph will be saved to the direcotry of the code
            errorwarning('Error:\nOutput directory not found, graph will be saved to the same directory as this code by default when this window is closed.\nCheck to see if you made a typo when specifying the output path.')
            titlestring = './' + plottitle+'.png'
            fig1.savefig(titlestring, bbox_inches="tight")

#works as scrollfunc2 above
def scrollfunc(event):
    global canvas
    canvas.configure(scrollregion=canvas.bbox("all"),width=400,height=500)

#works as canvascreate2 above
def canvascreate():
    global root, canvas, subframe, paramsframe
    paramsframe = Frame(root)
    paramsframe.grid(row=12,column=1)
    canvas=Canvas(paramsframe)
    subframe=Frame(canvas)
    myscrollbar=Scrollbar(paramsframe,orient="vertical",command=canvas.yview)
    canvas.configure(yscrollcommand=myscrollbar.set)
    
    myscrollbar.pack(side="right",fill="y")
    canvas.pack(side="top")
    canvas.create_window((200,0),window=subframe,anchor='n')
    subframe.bind("<Configure>",scrollfunc)
    return

#insertion point for the code and definition of the home window
if(__name__=='__main__'):
    root = Tk() #define root as main window
    root.title('General Graph Plotter') #set title of root
    #display some information about the code
    infolabel = Label(root,text='Welcome to the general graph plotter, this software gives you two plotting options: polynomial or custom law.').grid(row=0,column=0,columnspan=3,sticky='nsew')
    infolabel2 = Label(root,text='The polynomial option will fit data to up to any poisitive integer power of x you choose, it will give the coefficients of each power of x, the y intercept and the reduced chi squared value').grid(row=1,column=0,columnspan=3,sticky='nsew')
    infolabel3 = Label(root,text='The custom law will allow you to write your own law to try and fit the data to. This can be anything such as trigonometric functions or exponential functions.').grid(row=2,column=0,columnspan=3,sticky='nsew')
    infolabel4 = Label(root,text='For custom mode you need to pass at least one fitting parameter with guesses of a value. This is because the code will optimise the fit around these parameters to find the best equation of fit.').grid(row=3,column=0,columnspan=3,sticky='nsew')
    infolabel5 = Label(root,text='To see an example of this mode please select the "Custom Help" button below.').grid(row=4,column=0,columnspan=3)
    blankspace = Label(root,text='').grid(row=5,column=0,columnspan=3,sticky='nsew') #blank label to increase spacing of widgets
    
    dataframe = Frame(root) #create frame in root for datafile entries
    dataframe.grid(row=6,column=1)

    #label entry combos explaining you must enter the directory of the data file and an entry to do so
    path = StringVar()
    loadlabel = Label(dataframe,text='Data Directory + File, Can Be\n.txt or .csv: ',relief='solid').grid(row=0,column=0,sticky='nsew')
    dataentry = Entry(dataframe,textvariable=path,relief='solid').grid(row=0,column=1,sticky='nsew')
    infolabel6 = Label(dataframe,text='Please paste path to comma\nseperated data in format x,y,y_error',relief='solid').grid(row=0,column=2,sticky='nsew')
     #label entry combos explaining you must enter the directory to output the file to and an entry to do so
    outpath = StringVar()
    outlabel = Label(dataframe,text='Graph Output Location: ',relief='solid').grid(row=1,column=0,sticky='nsew')
    outentry = Entry(dataframe,textvariable=outpath,relief='solid').grid(row=1,column=1,sticky='nsew')
    infolabel7 = Label(dataframe,text='Please paste path where graph should output,\nfile name will be same as graph title',relief='solid').grid(row=1,column=2,sticky='nsew')
    
    
    polyframe = Frame(root) #create frame for the polynomial entries and labels
    polyframe.grid(row=7,column=0,sticky='nsew')
    customframe = Frame(root) #create frame for the custom entries and labels
    customframe.grid(row=7,column=2,sticky='nsew')
    #set radiobuttons to choose between polynomial and custom law and default to polynomial
    v = IntVar()
    v.set(1)
    polybutton = Radiobutton(polyframe, text="Polynomial", variable=v, value=1,relief='solid').grid(row=0,column=0,columnspan=4,sticky='nsew')
    custombutton = Radiobutton(customframe, text="Custom Law", variable=v, value=2,relief='solid').grid(row=0,column=0,columnspan=4,sticky='nsew')
    
    #label and entry for highest power of polynomial
    powerlabel = Label(polyframe,text='Highest Power Of X: ',relief='solid').grid(row=1,column=0,columnspan=2,sticky='nsew')
    powervar = StringVar()
    powerentry = Entry(polyframe,textvariable=powervar,relief='solid').grid(row=1,column=2,sticky='nsew')
    
    #info about gradient
    gradientinfo = Label(polyframe,text='If graph is not linear and a gradient at a point is desired, type\nthe X ordinate of the point at which you\nwant the gradient.',relief='solid')
    gradientinfo.grid(row=2,column=0,columnspan=3,sticky='nsew')
    
    #label and entry for gradient of polynomial
    gradlabel = Label(polyframe,text='Gradient X Ordinate: ',relief='solid').grid(row=3,column=0,columnspan=2,sticky='nsew')
    gradvar = StringVar()
    gradentry = Entry(polyframe,textvariable=gradvar,relief='solid').grid(row=3,column=2,sticky='nsew')
    
    #label and entry for equation of custom law
    equationlabel = Label(customframe,text='Custom Equation: y = ',relief='solid').grid(row=1,column=0,columnspan=2,sticky='nsew')
    equationvar = StringVar()
    equationentry = Entry(customframe,textvariable=equationvar,relief='solid').grid(row=1,column=3,sticky='nsew')
    
    #label and entry for fitting parameters of custom law
    fittingparamslabel = Label(customframe,text='Fitting Parameters: ',relief='solid').grid(row=2,column=0,columnspan=2,sticky='nsew')
    fittingparamsvar = StringVar()
    fittingparamsentry = Entry(customframe,textvariable=fittingparamsvar,relief='solid').grid(row=2,column=3,sticky='nsew')
    
    #label and entry for fitting parameter guesses of custom law
    paramsguesslabel = Label(customframe,text='Parameter Guesses: ',relief='solid').grid(row=3,column=0,columnspan=2,sticky='nsew')
    paramsguessvar = StringVar()
    paramsguessentry = Entry(customframe,textvariable=paramsguessvar,relief='solid').grid(row=3,column=3,sticky='nsew')
    
    #button for help with custom mode that opens the help window
    customhelpbutton = Button(customframe, text='Custom Help',command = customhelp).grid(row=4,column=0,columnspan=4,sticky='nsew')
    
    #blank to increase spacing of widgets
    blanklabel = Label(dataframe,text='').grid(row=2,column=0,columnspan=3,sticky='nsew')
    
    #label and entry for graph title
    title = Label(dataframe,text='Graph Title: ',relief='solid').grid(row=3,column=0,sticky='nsew')
    titlevar = StringVar()
    titleentry = Entry(dataframe,textvariable=titlevar,relief='solid').grid(row=3,column=1,columnspan=2,sticky='nsew')
    
    #label and entry for graph x label
    xtitle = Label(dataframe,text='X-label: ',relief='solid').grid(row=4,column=0,sticky='nsew')
    xtitlevar = StringVar()
    xtitleentry = Entry(dataframe,textvariable=xtitlevar,relief='solid').grid(row=4,column=1,columnspan=2,sticky='nsew')
    
    #label and entry for graph y label
    ytitle = Label(dataframe,text='Y-label: ',relief='solid').grid(row=5,column=0,sticky='nsew')
    ytitlevar = StringVar()
    ytitleentry = Entry(dataframe,textvariable=ytitlevar,relief='solid').grid(row=5,column=1,columnspan=2,sticky='nsew')
    
    #blank label to increase spacing of widgets
    blanklabel = Label(dataframe,text='').grid(row=6,column=0,columnspan=3,sticky='nsew')
    
    #button that calls the plotting functoin to begin processing data entered labelled plot
    plotbutton = Button(root, text='Plot',command=lambda: plot(path,v,powervar,gradvar,equationvar,fittingparamsvar,paramsguessvar,xtitlevar,ytitlevar,titlevar,outpath))
    plotbutton.grid(row=9,column=1,sticky='nsew')
    
    #blank label to increase widget spacing
    blank = Label(root,text='').grid(row=11,column=1,sticky='nsew')
    
    root.update() #update root display
    root.mainloop() #display and loop the root window


