import numpy as np
import time
from scipy import stats
from bokeh.io import curdoc
from bokeh.layouts import gridplot, column, row, layout
from bokeh.plotting import figure, show, output_file, ColumnDataSource
from bokeh.models.widgets import Select, TextInput, Button, Slider, CheckboxGroup, RadioGroup, Div
from bokeh.models import LinearAxis, Range1d

def get_hist(bins, data, normalized):
    hist, edges = np.histogram(data, density=True, bins=bins)
    if normalized:
        hist = hist/float(5)
    return dict(top=hist, bottom=[0]*len(hist), left=edges[:-1], right=edges[1:] )

def get_sample_sd(sample):
    sample = np.array(sample)
    x_bar = np.mean(sample)
    return (np.sum((sample-x_bar)**2)/(len(sample)-1.))**0.5

def area_under_curve(support, y, y_base):
    xx = np.append(support, [support[-1], support[0]])
    yy = np.append(y,[y_base,y_base])
    return dict(x=xx, y=yy)

#GLOBALS
current_dist = stats.norm
samples_list=[]
means_list=[]
medians_list=[]
x_sample=[]
support = []
theoretical_mean = 0
theoretical_var = 0
t_selected = False
x_range = (0,0)
discrete = False
bootstrap_sd = 0
bootstrap_medians_mean = 0
cumul = []

# Instructions
div_select_distribution = Div(text="<h2>CLT - STEP 1:<br> </h2>\
● Select a distribution<br>\
● Adjust its basic parameters<br>\
The common distribution measures appear below the distribution dropbox\
", width=500, height=100)

div_population = Div(text="<h2>CLT - STEP 2:<br> </h2>\
● Select the sample size (number of observations)<br>\
● Press the button to take a sample of that size from the distribution selected in STEP 1<br><br>\
The blue line/stars (for continuous/discrete distributions) is the theoretical shape.<br>\
The pink circles are the individual observations, which can also be represented as a histogram.<br>\
The average value of these observations is shown with a purple dashed line. This value will become one of the purple circles in STEP 3.<br>\
[The green KDE (Kernel Density Estimators) is just one way to infer the shape of the distribution from the observations; it is not related to the Central Limit Theorem]<br>\
<br><br>\
<font color=red><strong><b>NOTE: Clicking on a legend entry hides/reveals the corresponding data.</b></strong></font> <br>\
This is useful when a plot seems overloaded.\
", width= 500, height =350)
div_CLT = Div(text="<h2>CLT - STEP 3:<br> </h2>\
● Select the number of times you want to repeat STEP 2<br>\
● Press the button to sample repeatedly from the distribution<br><br>\
Every purple circle is the mean of a single sample (corresponding the purple line in STEP 2)<br>\
The histogram of these sample means reveals that this new 'sampling distribution' is normal. This is the Central Limit Theorem.<br>\
<strong><b>Remember that the CLT is valid when the number of observations (selected in STEP 2) is sufficiently large, typically >30</b></strong><br><br>\
The black curve is the CLT predicted distribution<br><br>\
The horizontal blue/red lines are the Confidence Intervals for each of these sample means: blue when they correctly capture the theoretical median, red when they don't.<br>\
", width= 500, height =350)
div_bootstrap = Div(text="<h2>BOOTSTRAP SAMPLING<br> </h2>\
When the total number of available observations/number of samples is small or when other quantities (ie median) need to be calculated, one can repeatedly sample with replacement from a single sample<br><br>\
In this demo, we estimate the median using as 'original' sample the one selected in STEP 2, so the 'Select single sample...' button must have been pressed before running the bootstrap simulation.<br>\
", width= 500, height =350)
div_comparison = Div(text="<h2>BONUS<br> </h2>\
<h4>Compare and contrast the normal and t distributions</h4>\
For sample sizes / degrees of freedom larger than 30-40, the two are almost identical, but when the sample size is <10, the t-distribution has much heavier tails.<br>\
This difference becomes obvious when we calculate Confidence Intervals, like the ones we saw in STEP 3<br>\
", width= 500, height =400)

def update_div_mu_sigma(mu, sigma, median, mode):
    basic = "μ = "+"{0:.2f}".format(mu)+", σ = "+"{0:.2f}".format(sigma)
    if median is not None:
        basic += ", median = "+"{0:.2f}".format(median)
    if mode is not None:
        basic += ", mode = "+"{0:.2f}".format(mode)
    div_mu_sigma.text = "<font color=black>"+basic+"</font>"

# WIDGETS 1
select_distribution = Select(title="Distribution:", value="1", options=[("1","Normal (continuous)"), ("2","Exponential (continuous)"), ("3","Uniform (continuous)"),
    ("4","Beta (continuous)"), ("5","Poisson (discrete)")])
slider_param1 = Slider(start=-10, end=10, step=0.1, value=0, title="μ ")
slider_param2 = Slider(start=0, end=10, step=0.1, value=1, title="σ ")
div_mu_sigma = Div(text="<font color=black>μ = 0.00, σ = 1.00, median = 0.00, mode = 0.00</font>", width=300, height=50)

# WIDGETS 2
slider_sample_size_population = Slider(start=1, end=100, step=1, value=20, title="Sample size")
button_sample_population = Button(button_type="success")
slider_bandwidth = Slider(start=0.05, end=2, step=0.05, value=1, title="KDE bandwidth")

# WIDGETS 3
slider_repetitions_CLT = Slider(start=1, end=1000, step=1, value=100, title="Number of sampling repetitions")
button_sample_CLT = Button( button_type="success")
slider_CL_CLT = Slider(start=0, end=99, step=1, value=95, title="Confidence level [%]")
z_vs_t_group = RadioGroup(labels=["Normal distribution", "t-distribution"], active=0)

# WIDGETS 4
slider_sample_size_bootstrap = Slider(start=1000, end=100000, step=1000, value=10000, title="Number of bootstrap samples")
button_sample_bootstrap = Button(button_type="success")
slider_CL_bootstrap = Slider(start=0, end=100, step=1, value=95, title="Confidence level [%]")

################################################

p_population = figure(title="Population Distribution", background_fill_color="#E8DDCB")
p_population.xaxis.axis_label = 'x'
p_population.yaxis.axis_label = 'Pr(x)'

source_hist_1 = ColumnDataSource(data=dict(top=[],bottom=[], left=[], right=[]))
p_population.quad(top= 'top',bottom= 'bottom', left='left', right='right',  source=source_hist_1, fill_color="#8bbf6d", line_color="#033649", legend="histogram")

source_pdf = ColumnDataSource(data=dict(x=[], y=[]))
p_population.line(x='x', y='y', source=source_pdf, color="blue", line_width=3, legend="population pdf")

source_pmf = ColumnDataSource(data=dict(x=[], y=[]))
p_population.asterisk(x='x', y='y', source=source_pmf, size=10, color="blue", alpha=1, legend="population pmf")

source_sample = ColumnDataSource(data=dict(x=[], y=[]))
p_population.circle(x='x', y='y', source=source_sample, size=10, color="#ca5670", alpha=0.5, legend="sample observations")

source_kde = ColumnDataSource(data=dict(x=[], y=[]))
p_population.line(x='x', y='y', source=source_kde, color="green", line_width=2, legend="KDE")

source_mean_one = ColumnDataSource(data=dict(x=[]))
p_population.ray(x='x', y=0, source=source_mean_one, color="#ab62c0", line_dash='dashed', line_width=3, length=0, angle=90, angle_units="deg", legend="sample mean")

p_population.legend.click_policy="hide"

################################################

p_CLT = figure(title="Sampling distribution of the mean", background_fill_color="#E8DDCB")#,  x_range=(-5,5))
p_CLT.xaxis.axis_label = 'x'
p_CLT.yaxis.axis_label = 'Sample Index'

p_CLT.extra_y_ranges['foo'] = Range1d(0, 1)
p_CLT.add_layout(LinearAxis(y_range_name="foo", axis_label="Pr(x)"), 'right')

source_sample_means_hist = ColumnDataSource(data=dict(top=[],bottom=[], left=[], right=[]))
p_CLT.quad(top= 'top',bottom= 'bottom', left='left', right='right',  source=source_sample_means_hist, fill_color="#8bbf6d", line_color="#033649", y_range_name="foo", legend="histogram")

source_sample_means = ColumnDataSource(data=dict(x=[], y=[]))
p_CLT.circle(x='x', y='y', source=source_sample_means, size=5, color="#ab62c0", alpha=1, legend="sample mean")

source_conf_intervals = ColumnDataSource(data=dict(xs=[  ], ys=[  ], color = []))
p_CLT.multi_line(xs='xs', ys='ys', color = 'color', source=source_conf_intervals, line_width=1, legend="Conf. interval")

source_theoretical_mean = ColumnDataSource(data=dict(x=[]))
p_CLT.ray(x='x', y=0, source=source_theoretical_mean, color="black", line_dash='dashed', line_width=2, length=0, angle=90, angle_units="deg", legend="population mean")

source_theoretical_gaussian = ColumnDataSource(data=dict(x=[], y=[]))
p_CLT.line(x='x', y='y', source=source_theoretical_gaussian, color="black", line_width=2, y_range_name="foo", legend="CLT sample mean distribution")

p_CLT.legend.click_policy="hide"

################################################

p_bootstrap = figure(title="Bootstrap sampling distribution of the median", background_fill_color="#E8DDCB")
p_bootstrap.xaxis.axis_label = 'x'

source_hist_boot = ColumnDataSource(data=dict(top=[],bottom=[], left=[], right=[]))
p_bootstrap.quad(top= 'top',bottom= 'bottom', left='left', right='right',  source=source_hist_boot, fill_color="#8bbf6d", line_color="#033649", legend="histogram")

source_median= ColumnDataSource(data=dict(x=[]))
p_bootstrap.ray(x='x', y=0, source=source_median, color="black", line_dash='dashed', line_width=2, length=0, angle=90, angle_units="deg", legend="bootstrap median")

source_boot_conf_normal = ColumnDataSource(data=dict(x=[], y=[]))
p_bootstrap.patch(x='x', y='y', source=source_boot_conf_normal, color="#638ccc", alpha=0.3, line_width=0, legend="CI assuming normality")

source_boot_conf_percent = ColumnDataSource(data=dict(x=[], y=[]))
p_bootstrap.patch(x='x', y='y', source=source_boot_conf_percent, color="#ab62c0", alpha=0.3, line_width=0, legend="CI percentile")

source_cumul= ColumnDataSource(data=dict(x=[], y=[]))
p_bootstrap.line(x='x', y='y', source=source_cumul, color="#ab62c0", line_width=2, legend="CDF")

source_boot_normal= ColumnDataSource(data=dict(x=[], y=[]))
p_bootstrap.line(x='x', y='y', source=source_boot_normal, color="#638ccc", line_width=2, legend="Normal")

p_bootstrap.legend.click_policy="hide"

################################################

def update_distribution_type(attr, old, new):
    global current_dist, support, theoretical_mean, theoretical_var, x_range, discrete

    source_sample.data = dict(x=[], y=[])
    source_mean_one.data = dict(x=[] )
    source_kde.data = dict(x=[], y=[])
    source_hist_1.data = dict(top=[],bottom=[], left=[], right=[])

    if select_distribution.value=='1': #Gaussian
        slider_param1.title="μ"
        slider_param1.start=-10
        slider_param1.end=10
        slider_param2.title="σ"
        slider_param2.start=0.1
        slider_param2.end=10

        current_dist = stats.norm(float(slider_param1.value), scale=float(slider_param2.value))
        theoretical_mean = slider_param1.value
        theoretical_var = slider_param2.value**2
        theoretical_median = slider_param1.value
        theoretical_mode = slider_param1.value
    elif select_distribution.value=='2': #Exponential
        slider_param1.title="λ"
        slider_param1.start=0.1
        slider_param1.end=10
        if slider_param1.value<=0:
            slider_param1.value=0.1
        slider_param2.title=" "

        current_dist = stats.expon(loc=0, scale=1./slider_param1.value)
        theoretical_mean = 1./slider_param1.value
        theoretical_var = slider_param1.value**(-2)
        theoretical_median = np.log(2)/slider_param1.value
        theoretical_mode = 0
    elif select_distribution.value=='3': #Uniform
        slider_param1.title="min"
        slider_param1.start=-10
        slider_param1.end=10
        slider_param2.title="max"
        slider_param2.start=-10
        slider_param2.end=10

        current_dist = stats.uniform(loc=slider_param1.value, scale=slider_param2.value-slider_param1.value)
        theoretical_mean = 0.5*(slider_param2.value+slider_param1.value)
        theoretical_var = (slider_param2.value-slider_param1.value)**2/12.
        theoretical_median = 0.5*(slider_param2.value+slider_param1.value)
        theoretical_mode = None
    elif select_distribution.value=='4': #Beta
        slider_param1.title="α"
        slider_param1.start=0.1
        slider_param1.end=10
        slider_param2.title="β"
        slider_param2.start=0.1
        slider_param2.end=10
        if slider_param1.value<=0:
            slider_param1.value=0.1
        if slider_param2.value<=0:
            slider_param2.value=0.1

        a = slider_param1.value
        b = slider_param2.value
        current_dist = stats.beta(a=a, b=b)
        theoretical_mean = a/float(a+b)
        theoretical_var = a*b / float( ((a+b)**2) * (a+b+1) )
        theoretical_median = stats.beta.median(a=a, b=b)
        if a>1 and b>1:
            theoretical_mode=(a-1)/float(a+b-2)
        else:
            theoretical_mode = None
    elif select_distribution.value=='5': #Poisson
        slider_param1.title="λ"
        slider_param1.start=0.1
        slider_param1.end=10
        if slider_param1.value<=0:
            slider_param1.value=0.1
        slider_param2.title=""

        lam = slider_param1.value
        current_dist = stats.poisson(mu=lam)
        theoretical_mean = lam
        theoretical_var = lam
        theoretical_median = np.floor(lam+1./3-0.02/lam)
        theoretical_mode = np.floor(lam)

    update_div_mu_sigma(theoretical_mean, theoretical_var**0.5, theoretical_median, theoretical_mode)
    x_range = (current_dist.ppf(0.0001), current_dist.ppf(0.9999) )

    support = np.linspace(x_range[0], x_range[1], 200)
    if select_distribution.value in ['1', '2', '3', '4']: # CONTINUOUS
        discrete = False
        y = current_dist.pdf(support)
        source_pmf.data = dict(x=[], y=[])
        source_pdf.data = dict(x=support, y=y)
    else:
        discrete = True
        support_disc = np.arange(x_range[0], x_range[1])
        y = current_dist.pmf(support_disc)
        source_pdf.data = dict(x=[], y=[])
        source_pmf.data = dict(x=support_disc, y=y)

def take_single_sample_population():
    global x_sample
    sample_size = int(slider_sample_size_population.value)

    x_sample = current_dist.rvs(size=sample_size)
    y = [0]*sample_size
    source_sample.data = dict(x=x_sample, y=y)
    source_mean_one.data = dict(x=[np.mean(x_sample)] )

    bins = np.linspace(x_range[0]-0.1, x_range[1]+0.1, max(7, 2+ 5*int(x_range[1]- x_range[0]) ) )
    source_hist_1.data = get_hist(bins, x_sample, discrete)

    update_text_button_sample_bootstrap(0, 0, 0)
    draw_kde(0,0,0)

def draw_kde(attrname, old, new):
    y = np.sum(np.array([stats.norm.pdf(support, loc = i, scale=slider_bandwidth.value) for i in x_sample])/float(slider_sample_size_population.value), axis=0)
    source_kde.data = dict(x=support, y=y)

def update_conf_levels_boot(attrname, old, new):
    CL = slider_CL_bootstrap.value/100.
    if CL==1:
        z_t = bootstrap_medians_mean + 5
    else:
        z_t = stats.norm.ppf((1+CL)/2)

    a = (100 - slider_CL_bootstrap.value)/2.

    bins = np.linspace(x_range[0], x_range[1], 100 )
    bootstrap_size = int(slider_sample_size_bootstrap.value)
    hist, edges = np.histogram(medians_list, density=False, bins=bins)
    cumul = np.cumsum(hist)/float(bootstrap_size)
    xmin = np.percentile(medians_list, a)
    xmax = np.percentile(medians_list, 100-a)

    index_min = (np.abs(edges-xmin)).argmin()
    index_max = (np.abs(edges-xmax)).argmin()

    source_boot_conf_percent.data = area_under_curve(edges[index_min:index_max], cumul[index_min:index_max], 0)

    supp = np.linspace(bootstrap_medians_mean-z_t*bootstrap_sd, bootstrap_medians_mean+z_t*bootstrap_sd, 100)
    current_dist = stats.norm(loc=bootstrap_medians_mean, scale=bootstrap_sd)
    source_boot_conf_normal.data = area_under_curve(supp, current_dist.pdf(supp), 0)

def update_conf_levels(attrname, old, new):
    CL = slider_CL_CLT.value/100.

    if t_selected:
        z_t = -stats.t.ppf((1-CL)/2.,slider_sample_size_population.value-1 )
    else:
        z_t = stats.norm.ppf((1+CL)/2)

    xs=[]
    ys=[]
    color=[]

    for i in xrange(int(slider_repetitions_CLT.value)):
        sample = samples_list[i]
        mean = np.mean(sample)
        conf_int = z_t * get_sample_sd(sample) / (slider_sample_size_population.value)**0.5

        xs.append([mean-conf_int,mean+conf_int])
        ys.append([i,i])
        if mean+conf_int<theoretical_mean or mean-conf_int>theoretical_mean:
            color.append("red")
        else :
            color.append("blue")

    source_conf_intervals.data = dict(xs=xs, ys=ys, color=color)

def take_many_samples_for_CLT():
    global samples_list, means_list

    sample_size = int(slider_sample_size_population.value)

    source_sample_means.data = dict(x=[], y=[])
    source_conf_intervals.data = dict(xs=[], ys=[], color=[])
    source_theoretical_mean.data = dict(x=[theoretical_mean] )

    y1 = stats.norm.pdf(support, loc = theoretical_mean, scale=(theoretical_var/float(sample_size))**0.5)
    source_theoretical_gaussian.data = dict(x=support, y=y1)
    p_CLT.extra_y_ranges['foo'].end = 0.5*np.ceil(2*np.max(y1))

    samples_list=[]
    means_list = []
    bins = np.linspace(x_range[0], x_range[1], 100 )

    for i in xrange(int(slider_repetitions_CLT.value)):
        x_sample = current_dist.rvs(size=sample_size)
        samples_list.append(x_sample)
        new_mean = np.mean(x_sample)
        means_list.append(new_mean)

        source_sample_means.stream(dict(y=[i], x=[new_mean]))
        source_sample_means_hist.data = get_hist(bins, means_list, False)

        time.sleep(0.007)

def take_bootstrap_samples():
    global medians_list, bootstrap_sd, bootstrap_medians_mean, cumul

    source_boot_conf_percent.data = dict(x=[], y=[])
    source_boot_conf_normal.data = dict(x=[], y=[])

    sample_size = int(slider_sample_size_population.value)
    bootstrap_size = int(slider_sample_size_bootstrap.value)

    medians_list = []
    bins = np.linspace(x_range[0], x_range[1], 100 )
    for i in xrange(bootstrap_size):
        bootstrap_sample = np.random.choice(x_sample, sample_size, replace=True)
        new_median = np.median(bootstrap_sample)
        medians_list.append(new_median)

    bootstrap_medians_mean = np.mean(medians_list)
    bootstrap_sd = get_sample_sd(medians_list)

    support = np.linspace(x_range[0], x_range[1], 100)
    current_dist = stats.norm(loc=bootstrap_medians_mean, scale=bootstrap_sd)
    source_boot_normal.data = dict(x=support, y=current_dist.pdf(support))

    source_hist_boot.data = get_hist(bins, medians_list, False)
    hist, edges = np.histogram(medians_list, density=False, bins=bins)
    cumul = np.cumsum(hist)/float(bootstrap_size)
    source_cumul.data = dict(x=edges[:-1], y=cumul)
    source_median.data = dict(x=[bootstrap_medians_mean] )

def update_text_button_sample_CLT(attr, old, new):
    button_sample_CLT.label="Select "+str(int(slider_repetitions_CLT.value))+" samples of size "+str(int(slider_sample_size_population.value))

def update_text_button_sample_population(attr, old, new):
    button_sample_population.label="Select single sample of size "+str(int(slider_sample_size_population.value))

def update_text_button_sample_bootstrap(attr, old, new):
    button_sample_bootstrap.label="Select "+str(int(slider_sample_size_bootstrap.value))+" bootstrap samples of size "+str(int(slider_sample_size_population.value))

def update_zt(active):
    global t_selected
    t_selected = bool(active)
    update_conf_levels(0,0,0)

################################################
update_text_button_sample_CLT(0, 0, 0)
update_text_button_sample_population(0, 0, 0)
update_text_button_sample_bootstrap(0,0,0)
update_distribution_type(0,0,0)
################################################
select_distribution.on_change('value', update_distribution_type)

button_sample_population.on_click(take_single_sample_population)
button_sample_bootstrap.on_click(take_bootstrap_samples)
button_sample_CLT.on_click(take_many_samples_for_CLT)
z_vs_t_group.on_click(update_zt)

slider_bandwidth.on_change('value', draw_kde)
slider_CL_CLT.on_change('value', update_conf_levels)
slider_CL_bootstrap.on_change('value', update_conf_levels_boot)
slider_sample_size_population.on_change('value', update_text_button_sample_population)
slider_sample_size_population.on_change('value', update_text_button_sample_CLT)
slider_repetitions_CLT.on_change('value', update_text_button_sample_CLT)
slider_sample_size_bootstrap.on_change('value', update_text_button_sample_bootstrap)
for slider in [slider_param1, slider_param2]:
    slider.on_change('value', update_distribution_type)

################################################
p_comparison = figure(title="Normal vs t-distribution", background_fill_color="#E8DDCB", x_range=(-5,5))
p_comparison.xaxis.axis_label = 'x'
p_comparison.yaxis.axis_label = 'Pr(x)'

source_normal_pdf = ColumnDataSource(data=dict(x=[], y=[]))
p_comparison.line(x='x', y='y', source=source_normal_pdf, color="blue", line_width=2, legend="normal pdf")

source_t_pdf = ColumnDataSource(data=dict(x=[], y=[]))
p_comparison.line(x='x', y='y', source=source_t_pdf, color="red", line_width=1, legend="t-distribution pdf")

source_CI_normal = ColumnDataSource(data=dict(x=[], y=[]))
p_comparison.patch(x='x', y='y', source=source_CI_normal, color="blue", alpha=0.3, line_width=0, legend="normal CI")

source_CI_t = ColumnDataSource(data=dict(x=[],y = []))
p_comparison.patch(x='x', y='y', source=source_CI_t, color="red", alpha=0.3, line_width=0, legend="t CI")

p_comparison.legend.click_policy="hide"

# WIDGETS 5
slider_sample_size_comparison = Slider(start=2, end=100, step=1, value=20, title="Sample size")
slider_CL_comparison = Slider(start=0, end=100, step=1, value=0, title="Confidence level [%]")

def update_comparison(attrname, old, new):
    support_z_t = np.linspace(-5, 5, 200)

    current_dist = stats.norm(loc=0, scale=1)
    y = current_dist.pdf(support_z_t)
    source_normal_pdf.data = dict(x=support_z_t, y=y)
    z = min(5, stats.norm.ppf(0.5*(1+slider_CL_comparison.value/100.)))
    x_range_normal = np.linspace(-z, z, 100)
    source_CI_normal.data = area_under_curve(x_range_normal, current_dist.pdf(x_range_normal), 0)

    current_dist = stats.t(df=slider_sample_size_comparison.value-1, loc=0, scale=1 )
    y = current_dist.pdf(support_z_t)
    source_t_pdf.data = dict(x=support_z_t, y=y)
    t = min(5, -stats.t.ppf(0.5*(1-slider_CL_comparison.value/100.), slider_sample_size_comparison.value-1 ) )
    x_range_t = np.linspace(-t, t, 100)
    source_CI_t.data = area_under_curve(x_range_t, current_dist.pdf(x_range_t), 0)

slider_sample_size_comparison.on_change('value', update_comparison)
slider_CL_comparison.on_change('value', update_comparison)

update_comparison(0,0,0)
################################################
mainLayout = column(
            row(column(select_distribution, div_mu_sigma), column(slider_param1, slider_param2),  div_select_distribution),
            row(p_population, column(div_population, slider_sample_size_population, button_sample_population, slider_bandwidth) ),
            row(p_CLT, column(div_CLT, slider_repetitions_CLT, button_sample_CLT,  row(slider_CL_CLT, z_vs_t_group)) ),
            row(p_bootstrap, column(div_bootstrap, slider_sample_size_bootstrap, button_sample_bootstrap, slider_CL_bootstrap)),
            row(p_comparison, column(div_comparison, slider_sample_size_comparison, slider_CL_comparison)),
    name='mainLayout')
curdoc().add_root(mainLayout)
