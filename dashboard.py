#%%imports
import dash
import flask
from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc
import dash_table as dtable
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import random
import itertools
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from functools import partial
from tqdm import tqdm
import json
#%%generating data
df = pd.read_csv("PowerBI.csv", encoding='latin')
for x in range(len(df.columns)):
    if 'Date' in df.columns[x]:
        index = 0
        for i in df[df.columns[x]]:
            if df[df.columns[x]][index]:
                df[df.columns[x]][index] = pd.Timestamp(i)
            index+=1
df['oldmonth'] = 'a'
df['oldquarter'] = 'Q'
#df.loc[(df['Repeat Observation'] == 'Yes') & (df['Status'] == 'Open'), ['Status']] = 'Repeat'
for x in range(len(df.Status)): 
    df.Status[x] = random.choice(['Open', 'Open', 'Open', 'Closed', 'Repeat'])
    df.oldmonth[x] = random.choice(['Jan 2020', 'Feb 2020', 'Mar 2020', 'Apr 2020', 'May 2020', 'Jun 2020', 'Jul 2020', 'Aug 2020', 'Sep 2020', 'Oct 2020', 'Nov 2020', 'Dec 2020'])
    df.Location[x] = random.choice(['Kandivali west', 'Borivali west', 'Malad west', 'Kandivali east', 'Borivali east', 'Malad east'])
    df.Brand[x] = random.choice(['A', 'B', 'C', 'D'])
    df.oldquarter[x] = 'Q'+str(pd.Timestamp(df.oldmonth[x]).quarter)
geolocator = Nominatim(user_agent="dashboardtrial")
geocode = RateLimiter(partial(geolocator.geocode, language = 'en'), min_delay_seconds=1)
reverse = RateLimiter(partial(geolocator.reverse, language = 'en'), min_delay_seconds=1)
geodictlat = {}
geodictlon = {}
geodictcity = {}
geouniques = df.Location.unique()
for x in tqdm(range(len(geouniques))) :
    location = geocode(geouniques[x])
    if location:
        geodictlat[geouniques[x]] = location.latitude
        geodictlon[geouniques[x]] = location.longitude
        locrev = reverse(str(location.latitude)+','+str(location.longitude))
        if locrev.raw.get('address').get('city'):
            geodictcity[geouniques[x]] = locrev.raw.get('address').get('city')
valuescity = [geodictcity.get(i, str('not found')) for i in geouniques]
df['city'] = np.select([df.Location == geouniques[x] for x in range(len(geouniques))], valuescity)
df['status1'] = np.where(df.Status == 'Repeat', 'Open', df.Status)
df['Region'].fillna('none', inplace = True)
config = {'displaylogo': False}
#%%roughprints

#%%
def mapplot(df):
    geouniques = df.Location.unique()
    valuessize = []
    for x in range(len(geouniques)):
        valuessize.append(df.Location[df.Location == geouniques[x]].count())
    valueslat = [geodictlat.get(i) for i in geouniques]
    valueslon = [geodictlon.get(i) for i in geouniques]
    geodf = pd.DataFrame(dict(Location = geouniques, lat = valueslat, lon = valueslon, sizeref = valuessize))
    scalefactor = 10**(str(geodf.sizeref.mean()).find('.') - 2.5)
    fig = go.Figure(go.Scattermapbox(lat=geodf.lat, lon=geodf.lon, marker_size=geodf.sizeref, marker_sizeref = scalefactor, marker_opacity = 0.7, marker_color = '#42836d', hovertext = geodf.Location, hoverinfo = 'text', customdata = geodf.Location, selected = dict(marker_color = '#42836d', marker_opacity = 0.7), unselected = dict(marker_color = 'mediumaquamarine', marker_opacity = 0.7)))
    fig.update_layout(selectionrevision = 'none', mapbox_style="open-street-map", mapbox_center=go.layout.mapbox.Center(lat=geodf.lat.mean(), lon=geodf.lon.mean()) , mapbox_zoom = 3, hovermode = 'closest', clickmode='event+select', title = dict(text = 'Observations by Location', xanchor = 'left', yanchor = 'top'))
    return fig
#%%ribbonplot
def ribbonplot(df):
    data = []
    xtxt = list(df.oldmonth.unique())
    xtxt.sort(key = lambda date: pd.Timestamp(date))
    xval = []
    xval_scatter = []
    ylabel = ['Open', 'Closed', 'Repeat']
    temp_open_scatter_y1 = []
    xtxtlen = len(xtxt)
    for i in range(1,xtxtlen+1):
        xval.append(i)
        if i == 1:
            xval_scatter.extend([x for x in np.linspace(i-0.20, i+0.35, num=int(0.55/0.06))])
        elif i == xtxtlen:
            xval_scatter.extend([x for x in np.linspace(i-0.35, i+0.20, num=int(0.55/0.06))])
        else:
            xval_scatter.extend([x for x in np.linspace(i-0.35, i+0.35, num=int(0.7/0.06))])
        temp_open_scatter_y1.append(0) 
    temp_open_scatter_y0 = list(temp_open_scatter_y1)
    temp_closed_scatter_y0 = list(temp_open_scatter_y1)
    temp_repeat_scatter_y0 = list(temp_open_scatter_y1)
    temp_closed_scatter_y1 = list(temp_open_scatter_y1)
    temp_repeat_scatter_y1 = list(temp_open_scatter_y1)
    xcount = 0
    for xt,xv in zip(xtxt,xval):
        ybar = [df.Status[(df.oldmonth == xt) & (df.Status == y)].count() for y in ylabel]
        sortedy = sorted(zip(ybar,ylabel))
        ybar = [y for y,_ in sortedy]
        ylabel = [y for _,y in sortedy]
        count = 0
        openbardone = False
        closedbardone = False
        repeatbardone = False
        for i in ylabel:
            if(i == 'Open'):
                data.append(go.Bar(name = str(i), marker_color = 'grey', showlegend = False, legendgroup = str(i), x = [xv], y = [ybar[count]], width = 0.4, customdata = [{'date': xt, 'status': i}]))
                temp_open_scatter_y1[xcount] = temp_open_scatter_y0[xcount] + ybar[count]
                openbardone = True
                if(closedbardone == False):
                    temp_closed_scatter_y0[xcount]+=ybar[count]
                if(repeatbardone == False):
                    temp_repeat_scatter_y0[xcount]+=ybar[count]
            elif(i == 'Closed'):
                data.append(go.Bar(name = str(i), marker_color = 'mediumaquamarine', showlegend = False, legendgroup = str(i), x = [xv], y = [ybar[count]], width = 0.4, customdata = [{'date': xt, 'status': i}]))
                temp_closed_scatter_y1[xcount] = temp_closed_scatter_y0[xcount] + ybar[count]
                closedbardone = True
                if(openbardone == False):
                    temp_open_scatter_y0[xcount]+=ybar[count]
                if(repeatbardone == False):
                    temp_repeat_scatter_y0[xcount]+=ybar[count]
            else:
                data.append(go.Bar(name = str(i), marker_color = 'tomato', showlegend = False, legendgroup = str(i), x = [xv], y = [ybar[count]], width = 0.4, customdata = [{'date': xt, 'status': i}]))
                temp_repeat_scatter_y1[xcount] = temp_repeat_scatter_y0[xcount] + ybar[count]
                repeatbardone = True
                if(openbardone == False):
                    temp_open_scatter_y0[xcount]+=ybar[count]
                if(closedbardone == False):
                    temp_closed_scatter_y0[xcount]+=ybar[count]
            count+=1
        xcount+=1
    data.append(go.Bar(name = 'Open', marker_color = 'grey', hoverinfo = 'skip', legendgroup = 'Open', x = [xval[0]], y = [0], customdata = ['ignore']))
    data.append(go.Bar(name = 'Closed', marker_color = 'green', hoverinfo = 'skip', legendgroup = 'Closed', x = [xval[0]], y = [0], customdata = ['ignore']))
    data.append(go.Bar(name = 'Repeat', marker_color = 'red', hoverinfo = 'skip', legendgroup = 'Repeat', x = [xval[0]], y = [0], customdata = ['ignore']))
    if xtxtlen > 1:
        open_scatter_y0 = []
        open_scatter_y1 = []
        closed_scatter_y0 = []
        closed_scatter_y1 = []
        repeat_scatter_y0 = []
        repeat_scatter_y1 = []
        index=1
        for oy0,oy1,cy0,cy1,ry0,ry1 in zip(temp_open_scatter_y0, temp_open_scatter_y1, temp_closed_scatter_y0, temp_closed_scatter_y1, temp_repeat_scatter_y0, temp_repeat_scatter_y1):
            if index == xtxtlen:
                for i in np.linspace(1-0.35, 1+0.2, num=int(0.55/0.06)):
                    open_scatter_y0.extend([oy0])
                    open_scatter_y1.extend([oy1])
                    closed_scatter_y0.extend([cy0])
                    closed_scatter_y1.extend([cy1])
                    repeat_scatter_y0.extend([ry0])
                    repeat_scatter_y1.extend([ry1])
                lastoy0 = oy0
                lastoy1 = oy1
                lastcy0 = cy0
                lastcy1 = cy1
                lastry0 = ry0
                lastry1 = ry1
            elif index == 1:
                for i in np.linspace(1-0.2, 1+0.35, num=int(0.55/0.06)):
                    open_scatter_y0.extend([oy0])
                    open_scatter_y1.extend([oy1])
                    closed_scatter_y0.extend([cy0])
                    closed_scatter_y1.extend([cy1])
                    repeat_scatter_y0.extend([ry0])
                    repeat_scatter_y1.extend([ry1])
            else:
                for i in np.linspace(1-0.35, 1+0.35, num=int(0.7/0.06)):
                    open_scatter_y0.extend([oy0])
                    open_scatter_y1.extend([oy1])
                    closed_scatter_y0.extend([cy0])
                    closed_scatter_y1.extend([cy1])
                    repeat_scatter_y0.extend([ry0])
                    repeat_scatter_y1.extend([ry1])
            index+=1
        oxval_scatter = list(itertools.chain(xval_scatter, [xval_scatter[-1:-2:-1][0] for i in np.linspace(lastoy0,lastoy1,num=int((lastoy1-lastoy0)/0.1))], xval_scatter[::-1]))
        cxval_scatter = list(itertools.chain(xval_scatter, [xval_scatter[-1:-2:-1][0] for i in np.linspace(lastcy0,lastcy1,num=int((lastcy1-lastcy0)/0.1))], xval_scatter[::-1]))
        rxval_scatter = list(itertools.chain(xval_scatter, [xval_scatter[-1:-2:-1][0] for i in np.linspace(lastry0,lastry1,num=int((lastry1-lastry0)/0.1))], xval_scatter[::-1]))
        open_scatter_y = open_scatter_y1 + [i for i in np.linspace(lastoy1,lastoy0,num=int((lastoy1-lastoy0)/0.1))] + open_scatter_y0[::-1]
        closed_scatter_y = closed_scatter_y1 + [i for i in np.linspace(lastcy1,lastcy0,num=int((lastcy1-lastcy0)/0.1))] + closed_scatter_y0[::-1]
        repeat_scatter_y = repeat_scatter_y1 + [i for i in np.linspace(lastry1,lastry0,num=int((lastry1-lastry0)/0.1))] + repeat_scatter_y0[::-1]
        data.insert(0, go.Scatter(x=oxval_scatter, y=open_scatter_y, showlegend = False, legendgroup = 'Open', hoverinfo = 'skip', mode = 'lines', line_width = 0, line_color = 'grey', fill = 'toself', line_shape = 'spline', line_smoothing = 0.7))
        data.insert(0, go.Scatter(x=cxval_scatter, y=closed_scatter_y, showlegend = False, legendgroup = 'Closed', hoverinfo = 'skip', mode = 'lines', line_width = 0, line_color = 'mediumaquamarine', fill = 'toself', line_shape = 'spline', line_smoothing = 0.7))
        data.insert(0, go.Scatter(x=rxval_scatter, y=repeat_scatter_y, showlegend = False, legendgroup = 'Repeat', hoverinfo = 'skip', mode = 'lines', line_width = 0, line_color = 'tomato', fill = 'toself', line_shape = 'spline', line_smoothing = 0.7))
    layout = dict(selectionrevision = 'none', dragmode = 'select', barmode = 'stack', xaxis_tickmode = 'array', xaxis_tickvals = xval, xaxis_ticktext = xtxt, xaxis_tickangle = -60, hovermode = 'closest', title = dict(text = 'Observations by Month', xanchor = 'left', yanchor = 'top'), clickmode='event+select')
    fig = go.Figure(data = data, layout = layout)
    return fig
#%%
def citybarplot(df):
    valuescity = [i for i in geodictcity.values()]
    valuescity = np.unique(valuescity)
    valuessize = []
    height = len(valuescity)*45
    if height < 300:
        height = 300
    layout = dict(selectionrevision = 'none', dragmode = 'select', title = dict(text = 'Observations by City', xanchor = 'left', yanchor = 'top'), clickmode='event+select', height = height)
    for x in range(len(valuescity)):
        valuessize.append(df.city[df.city == valuescity[x]].count())
    fig = go.Figure(data = go.Bar(name = 'citybar', marker_color = 'darkslateblue', text = valuessize, textposition = 'auto', x = valuessize, y = valuescity, orientation = 'h', insidetextanchor = 'middle', insidetextfont = dict(color = 'white')), layout = layout)
    return fig
#%%

#%%
def auditeebarplot(df):
    auditees = df.Auditee.unique()
    height = len(df.Auditee.unique())*45
    if height < 300:
        height = 300
    layout = dict(selectionrevision = 'none', dragmode = 'select', height = height, title = dict(text = 'Observations by Auditee', xanchor = 'left', yanchor = 'top'), clickmode='event+select')
    valuessize = []
    for x in range(len(auditees)):
        valuessize.append(df.Auditee[df.Auditee == auditees[x]].count())
    fig = go.Figure(data = go.Bar(name = 'auditeebar', marker_color = 'seagreen', text = valuessize, x = valuessize, y = auditees, orientation = 'h', textposition = 'auto', insidetextanchor = 'middle', insidetextfont = dict(color = 'white')), layout = layout)
    return fig
#%%
def auditareabarplot(df):
    auditareas = df['Audit Area'].unique()
    height = len(df['Audit Area'].unique())*45
    if height < 300:
        height = 300
    layout = dict(selectionrevision = 'none', dragmode = 'select', height = height, title = dict(text = 'Observations by Audit Area', xanchor = 'left', yanchor = 'top'), clickmode='event+select')
    valuessize = []
    for x in range(len(auditareas)):
        valuessize.append(df['Audit Area'][df['Audit Area'] == auditareas[x]].count())
    fig = go.Figure(data = go.Bar(name = 'auditareabar', marker_color = 'gray', text = valuessize, x = valuessize, y = auditareas, orientation = 'h', textposition = 'auto', insidetextanchor = 'middle', insidetextfont = dict(color = 'white')), layout = layout)
    return fig
#%%
def table(df):
    regions = ['North', 'South', 'East', 'West', 'none']
    data = []
    for i in range(len(regions)):
        openobs = df.status1[(df.status1 == 'Open') & (df.Region == regions[i])].count()
        closedobs = df.status1[(df.status1 == 'Closed') & (df.Region == regions[i])].count()
        total = openobs+closedobs
        data.append({'id': regions[i], 'Region': regions[i], 'Total': total, 'Open': openobs, 'Closed': closedobs})
    return dtable.DataTable(id = 'table', data = data, columns = [{'name': i, 'id': i} for i in ['Region', 'Total', 'Open', 'Closed']], style_header = dict(backgroundColor = 'rgb(65,96,119)', fontWeight = 'bold', color = 'white', textAlign = 'center'), style_cell = dict(textAlign = 'center'), row_selectable = 'multi', sort_action = 'native', persistence = True, persisted_props = ['selected_rows', 'sort_by'])
#%%
def piechart(df):
    brands = df.Brand.unique()
    brands.sort()
    data = []
    for i in range(len(brands)):
        data.append(df.status1[(df.status1 == 'Open') & (df.Brand == brands[i])].count() + df.status1[(df.status1 == 'Closed') & (df.Brand == brands[i])].count())
    fig = px.pie(values=data, names=brands, title='Observations by Brands')
    fig.update_layout(clickmode = 'event+select')
    return fig
#%%
def card(df):
   return [html.Div([html.Pre(str(df['Observation Type'][df['Observation Type'] == category].count())),
            html.Pre(category)],
            id = category,
            className = 'card') for category in ['Critical', 'Important', 'Essential', 'Routine', 'Repeat', 'Breached']]
#%%
def main(df, selector):
    switcher = {1: [{'label' : i, 'value' : i} for i in df.SBU.unique()],
                2: [{'label' : i, 'value' : i} for i in df.Location.unique()],
                3: [{'label' : i, 'value' : i} for i in df.oldquarter.unique()],
                4: [{'label' : i, 'value' : i} for i in df.Department.unique()],
                5: table(df),
                6: card(df),
                7: mapplot(df),
                8: auditeebarplot(df),
                9: citybarplot(df),
                10: auditareabarplot(df),
                12: piechart(df)}
    if selector == 11:
        return ribbonplot(df)
    return switcher.get(selector)
#%%
def lay():
    return html.Div([
    html.Div([
        html.B('ETrends Technologies'),
        html.Pre('SBU'),
        dcc.Dropdown(id = 'sbudd', multi = True, options = [{'label' : i, 'value' : i} for i in df.SBU.unique()], persistence = True),
        html.Pre('Location'),
        dcc.Dropdown(id = 'locdd', multi = True, options = [{'label' : i, 'value' : i} for i in df.Location.unique()], persistence = True),
        html.Pre('Quarter'),
        dcc.Dropdown(id = 'quarterdd', multi = True, options = [{'label' : i, 'value' : i} for i in df.oldquarter.unique()], persistence = True),
        html.Pre('Department'),
        dcc.Dropdown(id = 'departmentdd', multi = True, options = [{'label' : i, 'value' : i} for i in df.Department.unique()], persistence = True)
        ]),
    html.Br(),
    html.Br(),
    html.Div(id = 'tablediv',
        children = table(df),
        ),
    html.Br(),
    html.Br(),
    html.Div(id = 'carddiv',
        children = card(df)
        ),
    html.Br(),
    html.Br(),
    html.Div(
        dcc.Graph(id = 'map', figure = mapplot(df), config = config)
        ),
    html.Br(),
    html.Br(),
    html.Div(
        dcc.Graph(id = 'auditeebar', figure = auditeebarplot(df), config = config),
        style = dict(overflowY = 'scroll', height = '300px')
        ),
    html.Br(),
    html.Br(),
    html.Div(
        dcc.Graph(id = 'citybar', figure = citybarplot(df), config = config),
        style = dict(overflowY = 'scroll', height = '300px')
        ),
    html.Br(),
    html.Br(),
    html.Div(
        dcc.Graph(id = 'auditareabar', figure = auditareabarplot(df), config = config),
        style = dict(overflowY = 'scroll', height = '300px')
        ),
    html.Br(),
    html.Br(),
    html.Div(
        dcc.Graph(id = 'ribbon', figure = ribbonplot(df), config = config)
        ),
    html.Div(
        dcc.Graph(id = 'pie', figure = piechart(df), config = config)
        ),
    html.Div(
        html.Pre(id='dumps')
        )
    ])
#%%
server = flask.Flask(__name__)
app = dash.Dash(__name__, server = server)
app.layout =  lay

@app.callback([Output('sbudd', 'options'), Output('locdd', 'options'), Output('quarterdd', 'options'), Output('departmentdd', 'options'), Output('tablediv', 'children'), Output('carddiv', 'children'), Output('map', 'figure'), Output('auditeebar', 'figure'), Output('citybar', 'figure'), Output('auditareabar', 'figure'), Output('ribbon', 'figure'), Output('pie', 'figure')], [Input('sbudd', 'value'), Input('locdd', 'value'), Input('quarterdd', 'value'), Input('departmentdd', 'value'), Input('table', 'selected_row_ids'), Input('map', 'selectedData'), Input('auditeebar', 'selectedData'), Input('citybar', 'selectedData'), Input('auditareabar', 'selectedData'), Input('ribbon', 'selectedData')])
def crossfilter(sbu, loc, quarter, department, region, locmap, auditee, city, auditarea, datestatus):
    elementnum = {'sbudd': 1, 'locdd': 2, 'quarterdd': 3, 'departmentdd': 4, 'table': 5, 'carddiv': 6, 'map': 7, 'auditeebar': 8, 'citybar': 9, 'auditareabar': 10, 'ribbon': 11, 'pie': 12}
    trigger = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    dfnew = df
    returns = []
    if not trigger:
        for i in elementnum.values():
            returns.append(dash.no_update)
        returns = tuple(returns)
        return returns
    if sbu:
        dfnew = dfnew[dfnew.SBU.isin(sbu)]
    if loc:
        dfnew = dfnew[dfnew.Location.isin(loc)]
    if quarter:
        dfnew = dfnew[dfnew.oldquarter.isin(quarter)]
    if department:
        dfnew = dfnew[dfnew.Department.isin(department)]
    if region:
        dfnew = dfnew[dfnew.Region.isin(region)]
    if locmap:
        if locmap['points']:
            loc1 = []
            for i in range(len(locmap['points'])):
                loc1.append(locmap['points'][i]['customdata'])
            dfnew = dfnew[dfnew.Location.isin(loc1)]
    if auditee:    
        if auditee['points']:
            auditees = []
            for i in range(len(auditee['points'])):
                auditees.append(auditee['points'][i]['y'])
            dfnew = dfnew[dfnew.Auditee.isin(auditees)]
    if city:
        if city['points']:
            cities = []
            for i in range(len(city['points'])):
                cities.append(city['points'][i]['y'])
            dfnew = dfnew[dfnew.city.isin(cities)]
    if auditarea:
        if auditarea['points']:
            auditareas = []
            for i in range(len(auditarea['points'])):
                auditareas.append(auditarea['points'][i]['y'])
            dfnew = dfnew[dfnew['Audit Area'].isin(auditareas)]
    if datestatus:
        if datestatus['points']:
            date = []
            status = []
            for i in range(len(datestatus['points'])):
                if datestatus['points'][i]['customdata'] != 'ignore':
                    date.append(datestatus['points'][i]['customdata']['date'])
                    status.append(datestatus['points'][i]['customdata']['status'])
            dfnew = dfnew[(dfnew.oldmonth.isin(date)) & (dfnew.Status.isin(status))]
    for i in elementnum.values():
        if i == elementnum.get(trigger):
            returns.append(dash.no_update)
        else:
            returns.append(main(dfnew, i))
    returns = tuple(returns)
    return returns



if __name__ == '__main__':
    app.run_server()

#%%

    
