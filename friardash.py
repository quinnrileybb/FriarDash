import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# -----------------------------
# Load Data & Compute wOBA for Hitters
# -----------------------------

st.set_page_config(page_title="FriarDash")
df = pd.read_csv('malvern_5_8.csv')

# wOBA weights mapping
woba_weights = {
    'Fly Out': 0, 'Ground Out': 0, 'Line Out': 0, 'Pop Out': 0,
    'Strike Out': 0, 'Bunt Ground Out': 0, 'BB': 0.69, 'HBP': 0.72,
    '1B': 0.88, '2B': 1.247, '3B': 1.578, 'HR': 2.031
}
df['woba_value'] = df['AtBatResult'].map(woba_weights)

# -----------------------------
# Sidebar: Select View
# -----------------------------
view = st.sidebar.selectbox('Select View', ['Pitcher', 'Hitter'])

if view == 'Pitcher':
    # -----------------------------
    # Pitcher Section
    # -----------------------------
    pitchers_df = df[df['PitcherTeam'] == 'Malvern Prep']
    selected_pitcher = st.sidebar.selectbox('Select a Pitcher', pitchers_df['Pitcher'].unique())
    df_pitcher = pitchers_df[pitchers_df['Pitcher'] == selected_pitcher].copy()
    st.header(f'Pitcher: {selected_pitcher}')

    # Derived columns
    strike_zone = {'x_min': -0.83, 'x_max': 0.83, 'z_min': 1.5, 'z_max': 3.2}
    df_pitcher['Zone'] = ((df_pitcher['x'].between(strike_zone['x_min'], strike_zone['x_max'])) &
                           (df_pitcher['y'].between(strike_zone['z_min'], strike_zone['z_max']))).astype(int)
    whiff_desc = ['Strike Swing and Miss']
    swing_desc = ['Strike In Play', 'Strike Foul', 'Strike Swing and Miss']
    strike_desc = swing_desc + ['Strike Looking']
    df_pitcher['whiffs'] = df_pitcher['PitchResult'].isin(whiff_desc).astype(int)
    df_pitcher['swings'] = df_pitcher['PitchResult'].isin(swing_desc).astype(int)
    df_pitcher['strike_pitch'] = df_pitcher['PitchResult'].isin(strike_desc).astype(int)
    df_pitcher['BIP'] = (df_pitcher['PitchResult'] == 'Strike In Play').astype(int)
    df_pitcher['csw'] = df_pitcher['PitchResult'].isin(['Strike Swing and Miss', 'Strike Looking']).astype(int)

    # Run Value mapping
    rv_weights = {
        'HR': -1.3743, '3B': -1.0576, '2B': -0.7661, '1B': -0.4673,
        'Ball': -0.0637, 'HBP': -0.0637, 'Strike Foul': 0.0381,
        'Strike Looking': 0.0651, 'Strike Swing and Miss': 0.1181,
        'Fly Out': 0.1956, 'Ground Out': 0.1956
    }
    df_pitcher['PlayByPlay'] = df_pitcher['PitchResult']
    df_pitcher.loc[df_pitcher['PitchResult'] == 'Strike In Play', 'PlayByPlay'] = df_pitcher['AtBatResult']
    df_pitcher['RunValue'] = df_pitcher['PlayByPlay'].map(rv_weights)

    # Summary Metrics
    games = df_pitcher['Date'].nunique()
    outs_list = ['Fly Out','Ground Out','Line Out','Pop Out','Strike Out','Bunt Ground Out']
    df_pitcher['Out'] = df_pitcher['AtBatResult'].isin(outs_list).astype(int)
    total_outs = df_pitcher['Out'].sum()
    innings = (total_outs // 3) + ((total_outs % 3) / 3)
    total_pitches = len(df_pitcher)
    atbat = df_pitcher['AtBatResult'].dropna()
    bb_pct = (atbat == 'BB').sum() / len(atbat) * 100 if len(atbat) else np.nan
    k_pct = (atbat == 'Strike Out').sum() / len(atbat) * 100 if len(atbat) else np.nan
    first_df = df_pitcher.sort_values('PitchNo').groupby(['Batter','Date','Inning']).first().reset_index()
    fps_pct = first_df['PitchResult'].isin(strike_desc).sum() / len(first_df) * 100 if len(first_df) else np.nan
    summary_df = pd.DataFrame({
        'Games': [games], 'IP': [round(innings,1)],
        'BB%': [round(bb_pct,1)], 'K%': [round(k_pct,1)], 'FPS%': [round(fps_pct,1)]
    })

    styled_summary = summary_df.style.format({
        'Games': '{:.0f}',
        'IP': '{:.1f}',
        'BB%': '{:.1f}',
        'K%': '{:.1f}',
        'FPS%': '{:.1f}'
    })

    # Display styled summary table
    st.subheader('Summary Metrics')
    st.dataframe(styled_summary)


    # Detailed Pitch Metrics
    agg = df_pitcher.groupby('PitchType').agg(
        Count=('PitchResult','count'),
        MPH=('PitchVelo','mean'),
        TopMPH=('PitchVelo','max'),
        BIP=('BIP','sum'),
        Whiffs=('whiffs','sum'),
        Swings=('swings','sum'),
        ZoneRate=('Zone','mean'),
        StrikeRate=('strike_pitch','mean'),
        wOBAcon=('woba_value','mean'),
        RVsum=('RunValue','sum'),
        CSW=('csw','sum')
    ).reset_index()
    agg['Usage %'] = (agg['Count'] / total_pitches * 100).round(1)
    agg = agg.sort_values('Usage %', ascending=False)
    agg['Whiff%'] = np.where(agg['Swings']>0, (agg['Whiffs']/agg['Swings']*100).round(1), 0)
    agg['Strike %'] = (agg['StrikeRate']*100).round(1)
    agg['Zone %'] = (agg['ZoneRate']*100).round(1)
    agg['CSW%'] = (agg['CSW']/agg['Count']*100).round(1)
    agg['RV/100'] = (agg['RVsum']/agg['Count']*100).round(2)
    agg['TopMPH'] = agg['TopMPH'].round(0).astype(int)
    agg['MPH'] = agg['MPH'].round(1)
    dpi = agg.rename(columns={'PitchType':'Type'})[[
        'Type','Usage %','Count','MPH','TopMPH','BIP','Whiff%','Strike %','Zone %','wOBAcon','CSW%','RV/100'
    ]]
    total_row = {
        'Type':'Total','Usage %':100.0,'Count':total_pitches,'MPH':np.nan,
        'TopMPH':int(df_pitcher['PitchVelo'].max()), 'BIP':int(df_pitcher['BIP'].sum()),
        'Whiff%': round(df_pitcher['whiffs'].sum()/df_pitcher['swings'].sum()*100,1) if df_pitcher['swings'].sum() else 0,
        'Strike %': round(df_pitcher['strike_pitch'].mean()*100,1),
        'Zone %': round(df_pitcher['Zone'].mean()*100,1),
        'wOBAcon': round(df_pitcher['woba_value'].mean(),3),
        'CSW%': round(df_pitcher['csw'].sum()/total_pitches*100,1),
        'RV/100': round(df_pitcher['RunValue'].sum()/total_pitches*100,2)
    }
    detailed_df = pd.concat([dpi, pd.DataFrame([total_row])], ignore_index=True)
    st.subheader('Detailed Pitch Metrics')
    st.dataframe(detailed_df)

    # Heatmaps Section
    st.subheader('Heatmaps')
    xlim, ylim = (-1.5,1.5), (1.0,4.0)
    def draw_heatmap(data, title, cmap):
        fig, ax = plt.subplots(figsize=(4, 5))
    # Only attempt KDE if there are >1 unique x and y
        if not data.empty and data['x'].nunique() > 1 and data['y'].nunique() > 1:
            try:
                sns.kdeplot(
                    x=data['x'],
                    y=data['y'],
                    ax=ax,
                    fill=True,
                    alpha=0.7,
                    cmap=cmap,
                    bw_adjust=0.5,
                    levels=5,        # enforce fixed contour levels
                    thresh=0.05      # drop areas below 5% density
                )
            except ValueError:
            # fallback to hexbin if KDE fails
                ax.hexbin(
                    data['x'], data['y'],
                    gridsize=25,
                    cmap=cmap,
                    mincnt=1,
                    alpha=0.7
                )
        else:
        # too few points for KDE → scatter
            ax.scatter(data['x'], data['y'], s=20, color='grey', alpha=0.6)

    # draw the strike zone box
        rect = Rectangle(
            (strike_zone['x_min'], strike_zone['z_min']),
            strike_zone['x_max'] - strike_zone['x_min'],
            strike_zone['z_max'] - strike_zone['z_min'],
            fill=False, edgecolor='black', linewidth=2
        )
        ax.add_patch(rect)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(1.0, 4.0)
        ax.axis('off')
        ax.set_title(title, fontsize=12)
        return fig
    display_heatmap = lambda figs: [col.pyplot(fig) for col,fig in zip(st.columns(3), figs)]
    cats = ['Fast Ball','Breaking Ball','Change Up']
    st.markdown('**Overall Pitch Location**')
    display_heatmap([draw_heatmap(df_pitcher[df_pitcher['PitchType']==pt],pt,'Reds') for pt in cats])
    st.markdown('**Damage Heatmaps**')
    display_heatmap([draw_heatmap(df_pitcher[(df_pitcher['PitchType']==pt)&df_pitcher['AtBatResult'].isin(['1B','2B','3B','HR'])], f"{pt} Damage",'Reds') for pt in cats])
    st.markdown('**Whiff Heatmaps**')
    display_heatmap([draw_heatmap(df_pitcher[(df_pitcher['PitchType']==pt)&(df_pitcher['PitchResult']=='Strike Swing and Miss')], f"{pt} Whiffs",'Reds') for pt in cats])
    st.markdown('**Called Strikes Heatmaps**')
    display_heatmap([draw_heatmap(df_pitcher[(df_pitcher['PitchType']==pt)&(df_pitcher['PitchResult']=='Strike Looking')], f"{pt} Called Strikes",'Reds') for pt in cats])

    # Visuals Section
    st.subheader('Visuals')
    df_pitcher['TypeCount'] = df_pitcher.groupby(['Date','PitchType']).cumcount()+1
    bins = list(range(1, df_pitcher['TypeCount'].max()+1, 3)) + [df_pitcher['TypeCount'].max()+1]
    labels = [f"{i}-{i+2}" for i in bins[:-1]]
    df_pitcher['TypeGroup'] = pd.cut(df_pitcher['TypeCount'], bins=bins, labels=labels, right=False)
    vg = df_pitcher.groupby(['TypeGroup','PitchType'])['PitchVelo'].mean().reset_index()
    fig1,ax1 = plt.subplots(figsize=(6,4))
    for pt in vg['PitchType'].unique():
        dfg = vg[vg['PitchType']==pt]
        ax1.plot(dfg['TypeGroup'], dfg['PitchVelo'], marker='o', label=pt)
    ax1.set(xlabel='Type Count Group', ylabel='Avg Velocity (MPH)', title='Vel by Type Count')
    ax1.legend(); st.pyplot(fig1)

    df_pitcher['Date'] = pd.to_datetime(df_pitcher['Date'], format='%m/%d/%y')
    pivot = (
        df_pitcher
        .groupby(['Date','PitchType'])['PitchVelo']
        .mean()
        .reset_index()
        .sort_values('Date')
        .pivot(index='Date', columns='PitchType', values='PitchVelo')
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(6,4))
    for pt in pivot.columns:
        ax.plot(pivot.index, pivot[pt], marker='o', label=pt)
    ax.set(xlabel='Date', ylabel='Avg Velocity (MPH)', title='Vel by Date')
    ax.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.kdeplot(
        data=df_pitcher,
        x='PitchVelo',
        hue='PitchType',
        fill=True,
        common_norm=False,
        alpha=0.5,
        bw_adjust=0.5,
        ax=ax
    )
    ax.set(
        xlabel='Velocity (MPH)',
        ylabel='Density',
        title=f'{selected_pitcher} – Velocity Density by Pitch Type'
    )
    plt.tight_layout()
    st.pyplot(fig)


else:

    st.header('Hitter')

# Filter for Malvern Prep hitters
    df_hit = df[df['BatterTeam'] == 'Malvern Prep'].copy()

# Hitter selection dropdown
    selected_hitter = st.sidebar.selectbox('Select Hitter', df_hit['Batter'].unique())
    df_hitter = df_hit[df_hit['Batter'] == selected_hitter].copy()
    st.subheader(f'Hitter: {selected_hitter}')

# Compute Games and Plate Appearances
    games = df_hitter['Date'].nunique()
    pas = df_hitter['AtBatResult'].count()

# Compute basic rates
    k_pct  = df_hitter['AtBatResult'].eq('Strike Out').sum() / pas * 100 if pas else np.nan
    bb_pct = df_hitter['AtBatResult'].eq('BB').sum()        / pas * 100 if pas else np.nan

# Build and display first summary table
    summary_df = pd.DataFrame([{
        'Games': games,
        'PAs':    pas,
        'K%':     round(k_pct,1),
        'BB%':    round(bb_pct,1)
    }])
    styled_summary = summary_df.style.format({
        'Games': '{:.0f}',
        'PAs':   '{:.0f}',
        'K%':    '{:.1f}',
        'BB%':   '{:.1f}'
    })
    st.subheader('Summary Metrics')
    st.dataframe(styled_summary)

# -----------------------------
# Advanced Plate Discipline Breakdown
# -----------------------------

# Define strike zone and descriptions
    hit_list = ['1B', '2B', '3B', 'HR']

# 1) filter out NotTracked
    df_h = df_hitter[df_hitter['PitchType'] != 'NotTracked'].copy()
    strike_zone = {'x_min': -0.83, 'x_max': 0.83, 'z_min': 1.5, 'z_max': 3.2}
    whiff_desc  = ['Strike Swing and Miss']
    swing_desc  = ['Strike In Play', 'Strike Foul', 'Strike Swing and Miss']

# Flags on each pitch
    df_hitter['Zone']   = ((df_hitter['x'].between(strike_zone['x_min'], strike_zone['x_max'])) &
                       (df_hitter['y'].between(strike_zone['z_min'], strike_zone['z_max']))).astype(int)
    df_hitter['swing']  = df_hitter['PitchResult'].isin(swing_desc).astype(int)
    df_hitter['whiff']  = df_hitter['PitchResult'].isin(whiff_desc).astype(int)
    df_hitter['foul']   = (df_hitter['PitchResult'] == 'Strike Foul').astype(int)
    df_hitter['called'] = (df_hitter['PitchResult'] == 'Strike Looking').astype(int)
    df_hitter['BIP']    = (df_hitter['PitchResult'] == 'Strike In Play').astype(int)

# Filter out NotTracked
    fh = df_hitter[df_hitter['PitchType'] != 'NotTracked'].copy()

# Define categories: each pitch type + velocity buckets + total
   # 1) make sure we keep NotTracked pitches
    df_all = df_hitter.copy()  

# 2) define categories in the exact order you want
    categories = [
        'Fast Ball','Breaking Ball','Change Up',
        '80+','83+','86+','BB 69+','NotTracked',
        '0-0','1-0','0-1','Behind','Ahead','Even','2 Strikes'
    ]

# 3) helper as before
    def compute_metrics(sub):
        n = len(sub)
        bip = sub['BIP'].sum()
        bcon = sub.loc[sub['BIP']==1, 'AtBatResult'].isin(hit_list).sum() / bip if bip else np.nan
        wbc = sub.loc[sub['PitchResult']=='Strike In Play','woba_value'].mean() if n else np.nan
        sw = sub['swing'].sum()
        sw_pct = sw/n*100 if n else np.nan
        z_pct = sub.loc[sub['Zone']==1,'swing'].sum() / (sub['Zone']==1).sum()*100 if (sub['Zone']==1).sum() else np.nan
        wh_pct = sub['whiff'].sum()/sw*100 if sw else np.nan
        cs_pct = sub['called'].sum()/n*100 if n else np.nan
        foul_pct = sub['foul'].sum()/n*100 if n else np.nan
        avg_velo = sub['PitchVelo'].mean() if n else np.nan
        return {
            'Count':        n,
            'BIP':          bip,
            'BAcon':        round(bcon,3),
            'wOBAcon':      round(wbc,3),
            'Swing%':       round(sw_pct,1),
            'Z-Swing%':     round(z_pct,1),
            'Whiff%':       round(wh_pct,1),
            'Foul%':        round(foul_pct,1),
            'Called%':      round(cs_pct,1),
            'Avg Velo':     round(avg_velo,1)
        }

# 4) build the breakdown dict
    breakdown = {}
    for cat in categories:
        if cat in ['Fast Ball','Breaking Ball','Change Up']:
            sub = df_all[df_all['PitchType']==cat]
        elif cat=='80+':
            sub = df_all[df_all['PitchVelo']>=80]
        elif cat=='83+':
            sub = df_all[df_all['PitchVelo']>=83]
        elif cat=='86+':
            sub = df_all[df_all['PitchVelo']>=86]
        elif cat=='BB 69+':
            sub = df_all[(df_all['PitchType']=='Breaking Ball')&(df_all['PitchVelo']>=69)]
        elif cat=='NotTracked':
            sub = df_all[df_all['PitchType']=='NotTracked']
        elif cat=='0-0':
            sub = df_all[(df_all['Balls']==0)&(df_all['Strikes']==0)]
        elif cat=='1-0':
            sub = df_all[(df_all['Balls']==1)&(df_all['Strikes']==0)]
        elif cat=='0-1':
            sub = df_all[(df_all['Balls']==0)&(df_all['Strikes']==1)]
        elif cat=='Early Count':
            sub = df_all[(df_all['Balls']<=1)&(df_all['Strikes']<=1)]
        elif cat=='Behind':
            sub = df_all[df_all['Balls']<df_all['Strikes']]
        elif cat=='Ahead':
            sub = df_all[df_all['Balls']>df_all['Strikes']]
        elif cat=='Even':
            sub = df_all[df_all['Balls']==df_all['Strikes']]
        elif cat=='2 Strikes':
            sub = df_all[df_all['Strikes']==2]
        breakdown[cat] = compute_metrics(sub)

# 5) make the DataFrame and force the row order
    df_break = pd.DataFrame.from_dict(breakdown, orient='index')
    df_break.index.name = 'Category'
    df_break = df_break.reindex(categories)

# 6) display in Streamlit
    st.subheader('Plate Discipline & Velocity Breakdown')
    st.dataframe(df_break.style.format({
        'Count':'{:.0f}','BIP':'{:.0f}','BAcon':'{:.3f}','wOBAcon':'{:.3f}',
        'Swing%':'{:.1f}','Z-Swing%':'{:.1f}',
        'Whiff%':'{:.1f}','Called%':'{:.1f}', 'Foul%':'{:.1f}', 'Avg Velo':'{:.1f}'
    }))

    # -----------------------------
# Hitter Heatmaps
# -----------------------------
# strike‐zone and plotting limits (reuse from earlier)
    strike_zone = {'x_min': -0.83, 'x_max': 0.83, 'z_min': 1.5, 'z_max': 3.2}
    xlim, ylim = (-1.5, 1.5), (1.0, 4.0)

    from matplotlib.patches import Rectangle

    def draw_heatmap(data, title, cmap):
        fig, ax = plt.subplots(figsize=(4, 5))
    # only attempt KDE if we have at least 2 unique points in each dimension
        if not data.empty and data['x'].nunique() > 1 and data['y'].nunique() > 1:
            try:
                sns.kdeplot(
                    x=data['x'],
                    y=data['y'],
                    ax=ax,
                    fill=True,
                    alpha=0.7,
                    cmap=cmap,
                    bw_adjust=0.5,
                # you can explicitly set levels too:
                    levels=5,
                    thresh=0.05
                )
            except ValueError:
            # fallback to a 2D hexbin if KDE failed
                hb = ax.hexbin(
                    data['x'], data['y'],
                    gridsize=25,
                    cmap=cmap,
                    mincnt=1,
                    alpha=0.7
                )
        else:
        # too few unique points for KDE → just scatter
            ax.scatter(data['x'], data['y'], s=20, color='grey', alpha=0.6)

    # draw the strike zone
        rect = Rectangle(
            (strike_zone['x_min'], strike_zone['z_min']),
            strike_zone['x_max'] - strike_zone['x_min'],
            strike_zone['z_max'] - strike_zone['z_min'],
            fill=False, edgecolor='black', linewidth=2
        )
        ax.add_patch(rect)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(1.0, 4.0)
        ax.axis('off')
        ax.set_title(title, fontsize=12)
        return fig

# the three pitch‐type categories in your data
    categories = ['Fast Ball', 'Breaking Ball', 'Change Up']

# four outcome filters
    outcomes = {
        'Swings':        df_hitter['swing'] == 1,
        'Whiffs':        df_hitter['whiff'] == 1,
        'Damage':        df_hitter['AtBatResult'].isin(['1B','2B','3B','HR']),
        'Called Strikes': df_hitter['PitchResult'] == 'Strike Looking'
    }

    for outcome_name, mask in outcomes.items():
        st.markdown(f'**{outcome_name} Heatmaps**')
        cols = st.columns(len(categories))
        for pt, col in zip(categories, cols):
            subset = df_hitter[(df_hitter['PitchType'] == pt) & mask]
            fig = draw_heatmap(subset, f"{pt} {outcome_name}", 'Reds')
            col.pyplot(fig)
