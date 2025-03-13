## Web App with CWordTM Toolkit ##
# Last updated: 23-Nov-2024, 24-Dec-2024, 4-Jan-2025, 15-Jan-2025, 1-Feb-2025
# 3-Mar-2025: Add 'Scripture Browsing' (from v4), 4-Mar-2025 (0.5.2)
# 5-Mar-2025 (0.6.0) : Add 'Help' page, 6-Mar-2025
# 11-Mar-2025 (0.6.1): Update 'Comments' table
# 12-Mar-2025 (0.7.0): Add caching mechanisms
# 13-Mar-2025 (0.6.2): Two DBs: Bible_v4.db & Bible_Comment.db
##

# Dependencies
import streamlit as st
from streamlit import components
from contextlib import contextmanager, redirect_stdout
from io import StringIO, BytesIO
import numpy as np
import pandas as pd
import cwordtm
from cwordtm import *
import pyLDAvis
import matplotlib.pyplot as plt
import plotly
import plotly.io as pio
import sqlite3


bbc = "BBC News Train.csv"
web = "web.csv"
cuv = "cuv.csv"
ch_label = None

@st.cache_data
def load_data(file_path):
    df = util.load_word(file_path)
    return df

if 'web' not in st.session_state:
    st.session_state.web = None

if web != st.session_state.web:
    st.session_state.web = web
    st.session_state.data = load_data(web) 


st.set_page_config(layout="wide")

cat_short = ['tor', 'oth', 'ket', 'map', 'mip', 'gos', 'nth', 'pau', 'epi', 'apo']

categories = ['Torah (摩西五經)',
              'OT History (舊約歷史書)',
              'Ketuvim (詩歌智慧書)',
              'Major Prophets (大先知書)',
              'Minor Prophets (小先知書)',
              'Gospel (福音書)',
              'NT History (新約歷史書)',
              'Pauline Epistles (保羅書信)',
              'General Epistles (普通書信)',
              'Apocalypse (啟示文學)']

otbks = ['Genesis (創世記)',
         'Exodus (出埃及記)',
         'Leviticus (利未記)',
         'Numbers (民數記)',
         'Deuteronomy (申命記)',
         'Joshua (約書亞記)',
         'Judges (士師記)',
         'Ruth (路得記)',
         '1 Samuel (撒母耳記上)',
         '2 Samuel (撒母耳記下)',
         '1 Kings (列王紀上)',
         '2 Kings (列王紀下)',
         '1 Chronicles (歷代志上)',
         '2 Chronicles (歷代志下)',
         'Ezra (以斯拉記)',
         'Nehemiah (尼希米記)',
         'Esther (以斯帖記)',
         'Job (約伯記)',
         'Psalms (詩篇)',
         'Proverbs (箴言)',
         'Ecclesiastes (傳道書)',
         'Song of Solomon (雅歌)',
         'Isaiah (以賽亞書)',
         'Jeremiah (耶利米書)',
         'Lamentations (耶利米哀歌)',
         'Ezekiel (以西結書)',
         'Daniel (但以理書)',
         'Hosea (何西阿書)',
         'Joel (約珥書)',
         'Amos (阿摩司書)',
         'Obadiah (俄巴底亞書)',
         'Jonah (約拿書)',
         'Micah (彌迦書)',
         'Nahum (那鴻書)',
         'Habakkuk (哈巴谷書)',
         'Zephaniah (西番雅書)',
         'Haggai (哈該書)',
         'Zechariah (撒迦利亞書)'
         'Malachi (瑪拉基書)']

ntbks = ['Matthew (馬太福音)',
         'Mark (馬可福音)',
         'Luke (路加福音)',
         'John (約翰福音)',
         'Acts (使徒行傳)',
         'Romans (羅馬書)',
         '1 Corinthians (哥林多前書)',
         '2 Corinthians (哥林多後書)',
         'Galatians (加拉太書)',
         'Ephesians (以弗所書)',
         'Philippians (腓立比書)',
         'Colossians (歌羅西書)',
         '1 Thessalonians (帖撒羅尼迦前書)',
         '2 Thessalonians (帖撒羅尼迦後書)',
         '1 Timothy (提摩太前書)',
         '2 Timothy (提摩太後書)',
         'Titus (提多書)',
         'Philemon (腓利門書)',
         'Hebrews (希伯來書)',
         'James (雅各書)',
         '1 Peter (彼得前書)',
         '2 Peter (彼得後書)',
         '1 John (約翰壹書)',
         '2 John (約翰貳書)',
         '3 John (約翰參書)',
         'Jude (猶大書)',
         'Revelation (啟示錄)']


# Define background image and its CSS style
bg_img = '''
    <style>
    .stApp {
        background-image: url("https://i.imgur.com/4vMRFIv.png");
        background-size: 80%;
        background-position: bottom right;
        background-repeat: no-repeat;
    }
    </style>
'''

# Define global CSS styles
st.markdown(
    """
    <style>
    h1 {
        font-family: "Cambria";
        font-size: 40px;
        color: blue;
    }
    h2 {
        font-family: "Cambria";
        font-size: 24px;
        font-style: italic;
        color: blue;
    }
    p {
        font-family: "Cambria";
        font-size: 20px;
    }
    .stSidebar h2 {
        font-family: "Cambria";
        font-size: 28px;
        color: brown;
    }
    .stSidebar .stSelectbox > div > div {
        font-size: 18px;
        color: blue;
    }
    .stSelectbox > div > div {
        font-family: "Cambria";
        font-size: 16px;
        color: blue;
    }
   </style>
    """,
    unsafe_allow_html=True
)

@st.dialog("Help")
def help_page():
    st.markdown("""
        <style>
        div[role="dialog"] {
            width: 60%;
        }
        </style>""", unsafe_allow_html=True)

    help_content = ""
    with open('cwordtm_app_help.txt', 'r', encoding='utf-8') as file:
       for line in file:
           help_content += line

    st.text(help_content)


def change_label_style(label, font_size='16px', font_color='black', font_family='sans-serif'):
    html = f"""
    <script>
        var elems = window.parent.document.querySelectorAll('p');
        var elem = Array.from(elems).find(x => x.innerText == '{label}');
        elem.style.fontSize = '{font_size}';
        elem.style.color = '{font_color}';
        elem.style.fontFamily = '{font_family}';
    </script>
    """
    st.components.v1.html(html)

def reset_label(label):
    change_label_style(label, '20px', 'blue', 'Cambria')

@st.dialog("Source Code", width="large")
def source(code):
    st.markdown("""
        <style>
        div[role="dialog"] {
            width: 80%;
        }
        </style>""", unsafe_allow_html=True)
    st.code(code)

@st.dialog("Scripture Statistics")
def bible_stat(stat_df, cat_df):
    st.markdown("""
        <style>
        div[role="dialog"] {
            width: 50%;
        }
        </style>""", unsafe_allow_html=True)
    st.dataframe(stat_df)
    st.html("<h3>Bible Book Category Information</h3>")
    st.dataframe(cat_df)

@st.dialog("Summary", width="large")
def summary_out(summary, source_docs):
    st.markdown("""
        <style>
        div[role="dialog"] {
            width: 80%;
        }
        p {
            font-family: "Cambria";
            font-size: 18px;
            color: blue;
        }
        </style>""", unsafe_allow_html=True)
    st.html(f"<p>{summary}</p>")
    st.html("<h3>Source Documents</h3>")
    st.dataframe(source_docs)

@st.dialog("Word Cloud", width="large")
def wordcloud(fig):
    st.markdown("""
        <style>
        div[role="dialog"] {
            width: 80%;
        }
        </style>""", unsafe_allow_html=True)
    st.pyplot(fig)
    fig_buf = BytesIO()
    fig.savefig(fig_buf, format='png')
    fig_buf.seek(0)
    st.markdown("""
        <style>
        .stDownloadButton {
            color: blue;
            display: flex;
            justify-content: center;
            align-items: center;
        }
       </style>""", unsafe_allow_html=True)
    st.download_button(
        label="Save Word Cloud (PNG)",
        data=fig_buf,
        file_name="wordcloud.png",
        mime="image/png"
    )

@st.dialog("LDA Visualization", width="large")
def lda_viz(html):
    st.markdown("""
        <style>
        div[role="dialog"] {
            width: 90%;
        }
        </style>""", unsafe_allow_html=True)

    st.components.v1.html(html, width=1200, height=800)

    st.markdown("""
        <style>
        .stDownloadButton {
            color: blue;
            display: flex;
            justify-content: center;
            align-items: center;
        }
       </style>""", unsafe_allow_html=True)

    st.download_button(
        label="Download LDA Visualization",
        data=html,
        file_name="lda_viz.html",
        mime="text/html"
    )

@st.dialog("Topic Model Visualization", width="large")
def tm_viz(fig_title, fig):
    st.markdown("""
        <style>
        div[role="dialog"] {
            width: 90%;
        }
        </style>""", unsafe_allow_html=True)

    fig_buf = BytesIO()
    st.pyplot(fig)  # matplotlib.figure.Figure
    fig.savefig(fig_buf, format='png')
    fig_buf.seek(0)

    st.markdown("""
        <style>
        .stDownloadButton {
            color: blue;
            display: flex;
            justify-content: center;
            align-items: center;
        }
       </style>""", unsafe_allow_html=True)

    st.download_button(
        label=f"Download {fig_title}",
        data=fig_buf,
        file_name=f"{fig_title}.png",
        mime="image/png"
    )

@st.dialog("BERTopic Model Visualization", width="large")
def btm_viz(figs):
    st.markdown("""
        <style>
        .stDownloadButton {
            color: blue;
            display: flex;
            justify-content: center;
            align-items: center;
        }
       </style>""", unsafe_allow_html=True)

    for fig_title, fig in figs:
        st.plotly_chart(fig)
        fig_buf = pio.to_html(fig, full_html=True)
        fig_file = f"{fig_title}.html"
        fig_mime = "text/html"

        if st.download_button(
            label=f"Download {fig_title}",
            data=fig_buf,
            file_name=fig_file,
            mime=fig_mime
        ):
            st.success(f"{fig_title} downloaded successfully!")


def show_exegesis(opt, book, chap, verse):
    db_conn = sqlite3.connect("Bible_v4.db")
    book_df = pd.read_sql_query("select * from Bible_Books", db_conn)
    engS = book_df[book_df.BookS==book].iloc[0].EngS

    if opt == 'Book Exegesis':
        sql = f"select Content from Comments where EngS='{engS}' and BChap=0"
    else:  # Verse Exegesis
        sql = f"select Content from Comments where \
                    EngS='{engS}' and \
                    BChap<={chap} and BSec<={verse} and \
                    EChap>={chap} and ESec>={verse}"

    comm_conn = sqlite3.connect("Bible_Comment.db")
    comments = pd.read_sql_query(sql, comm_conn)
    if comments.empty:
        st.text("No exegesis found!")
    else:
    	#st.text(comments.iloc[-1].Content)
    	st.text(comments.iloc[0].Content)

@st.dialog("Scripture Browsing")
def scripture_browsing(scdf):
    st.markdown("""
        <style>
        div[role="dialog"] {
            width: 80%;
        }
        p {
            font-family: "Cambria";
            font-size: 20px;
            color: blue;
        }
        </style>""", unsafe_allow_html=True)

    header = "Scripture: " + scdf.iloc[0].book + " " + \
                str(scdf.iloc[0].chapter)
    st.header(header, divider="blue")
    verse_selected = st.dataframe(
        scdf[['verse', 'text']], 
        hide_index=True,
        column_config={
            "text": st.column_config.TextColumn(width=800),
        },
        on_select="rerun",
        selection_mode="single-row"
    )

    if verse_selected is not None:
        selected_row = verse_selected['selection']['rows']
        if selected_row != []:
            menu_label = "Choose an option:"
            context_opt = st.selectbox(
                menu_label,
                ['<select>', 'Book Exegesis', 'Verse Exegesis']
            )
            # change_label_style(menu_label, '20px', 'blue', 'Cambria')

            if context_opt != '<select>':
                show_exegesis(context_opt, scdf.iloc[0].book, scdf.iloc[0].chapter, selected_row[0]+1)


@contextmanager
def st_capture(output_func):
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string):
            ret = old_write(string)
            output_func(stdout.getvalue())
            return ret
        
        stdout.write = new_write
        yield

# Streamlit App Main
def main():
    global ch_label

    st.markdown(bg_img, unsafe_allow_html=True)
    # st.sidebar.header("User Input")

    # Top-level Selection Box
    top_sel_label = "Choose a task:"
    top_opts = ['<select>',
                'Show Module Code',
                'Scripture Statistics',
                'Quotes from OT',
                'Wordcloud',
                'Text Summary',
                'Topic Modeling',
                'Scripture Browsing']

    top_opt = st.sidebar.selectbox(top_sel_label, options=top_opts)
    opt = top_opts.index(top_opt)
    change_label_style(top_sel_label, '20px', 'blue', 'Cambria')

    wdf = st.session_state.data

    # Filter from Top-level Selection #
    if opt == 1:  # Show module code
        mod_label = "Choose a module:"
        modules = ["meta", "pivot", "quot", "ta", "tm", "util", "viz"]
        mod_sel = st.sidebar.selectbox(mod_label, options=modules)
        change_label_style(mod_label, '20px', 'blue', 'Cambria')
        
        bi_label = "Show function signature (0) / detailed code (1)?"
        bi_opts = (0, 1)
        bi_opt = st.sidebar.radio(bi_label, bi_opts, horizontal=True)
        change_label_style(bi_label, '20px', 'blue', 'Cambria')

    elif opt == 3:  # Quotes from OT
        lang_label = "English / Chinese Text?"
        bi_opts = ('English', 'Chinese')
        bi_opt = st.sidebar.radio(lang_label, bi_opts, horizontal=True)
        bi_opt = bi_opts.index(bi_opt)
        change_label_style(lang_label, '20px', 'blue', 'Cambria')

        ntbk_label = "Choose an NT Book:"
        ntbk_opt = st.sidebar.selectbox(ntbk_label, options=ntbks, 
                                        on_change=lambda: reset_label(ch_label))
        ntbk_opt = ntbks.index(ntbk_opt)
        change_label_style(ntbk_label, '20px', 'blue', 'Cambria')

        bdf = util.extract(wdf, book=ntbk_opt+40)
        ch_label = "Choose a chapter:"
        ch_opts = list(bdf.chapter.unique())
        ch_opt = st.sidebar.selectbox(ch_label, options=ch_opts)
        change_label_style(ch_label, '20px', 'blue', 'Cambria')

        thres_label = "Choose a matching threshold"
        thres_opts = list(np.arange(3, 10) / 10)
        thres_opt = st.sidebar.selectbox(thres_label, options=thres_opts, index=1)
        change_label_style(thres_label, '20px', 'blue', 'Cambria')

    elif opt in [4, 5, 6]:  # Dataset; En / Chi; Scripture Scope
        data_label = "Choose a dataset"
        ds_opts = ['BBC News', 'Holy Bible', 'Other Dataset']
        ds_opt = st.sidebar.selectbox(data_label, options=ds_opts)
        ds_opt = ds_opts.index(ds_opt)
        change_label_style(data_label, '20px', 'blue', 'Cambria')

        if ds_opt == 0:  # BBC News
            news_label = "Select a range of news articles:"
            options = list(range(1, 1491))
            news_range = st.sidebar.select_slider(news_label,
                                                  options=options,
                                                  value=(1, 500))
            change_label_style(news_label, '20px', 'blue', 'Cambria')
        else:
            lang_label = "English / Chinese Text?"
            bi_opts = ('English', 'Chinese')
            bi_opt = st.sidebar.radio(lang_label, bi_opts, horizontal=True)
            bi_opt = bi_opts.index(bi_opt)
            change_label_style(lang_label, '20px', 'blue', 'Cambria')

        if ds_opt == 1:  # Holy Bible
            scop_label = "Choose Sripture scope:"
            scop_opts = ['Whole Bible', 'OT Book', 'NT Book', 'Category']
            scop_opt = st.sidebar.selectbox(scop_label, options=scop_opts)
            scop_opt = scop_opts.index(scop_opt)
            change_label_style(scop_label, '20px', 'blue', 'Cambria')
           
            if scop_opt > 0:
                if scop_opt == 1:  # OT Book
                    cat_label = "Choose an OT Book:"
                    cat_opts = ['Whole OT'] + otbks
                elif scop_opt == 2:  # NT Book
                    cat_label = "Choose an NT Book:"
                    cat_opts = ['Whole NT'] + ntbks
                else:  # Category
                    cat_label = "Choose a category:"
                    cat_opts = categories

                cat_opt = st.sidebar.selectbox(cat_label, options=cat_opts,
                                               on_change=lambda: reset_label(ch_label))
                cat_opt = cat_opts.index(cat_opt)
                change_label_style(cat_label, '20px', 'blue', 'Cambria')

                if opt != 6 and scop_opt in [1, 2] and cat_opt > 0:
                    bdf = util.extract(wdf, book=(scop_opt-1)*39+cat_opt)
                    ch_label = "Choose a chapter:"
                    ch_opts = ['Whole Book'] + list(bdf.chapter.unique())
                    ch_opt = st.sidebar.selectbox(ch_label, options=ch_opts)
                    ch_opt = ch_opts.index(ch_opt)
                    change_label_style(ch_label, '20px', 'blue', 'Cambria')

        elif ds_opt == 2:  # Other Dataset
            oth_label = "Upload Data File"
            oth_file = st.sidebar.file_uploader(oth_label, type=["csv", "txt"])
            change_label_style(oth_label, '20px', 'blue', 'Cambria')
            oth_opt = oth_file.name if oth_file else 0

    if opt == 4:  # Wordcloud
        mask_label = "Choose a mask image for your wordcloud:"
        mask_opts = ['Nil', 'Heart' ,'Disc', 'Triangle', 'Arrow', 'Other']
        mask_opt = st.sidebar.selectbox(mask_label, options=mask_opts)
        mask_opt = mask_opts.index(mask_opt)
        change_label_style(mask_label, '20px', 'blue', 'Cambria')
        
        if mask_opt == 5:  # Other image mask
            img_label = "Upload Image Mask"
            img_file = st.sidebar.file_uploader(img_label,
                                    type=["png", "jpg", "jpeg"])
            change_label_style(img_label, '20px', 'blue', 'Cambria')
            mask_opt = img_file if img_file else 0
 
    if opt == 5:  # Text Summary
        sent_label = "Limit the length of source sentences:"
        sent_opts = list(np.arange(8, 21))
        sent_opt = st.sidebar.selectbox(sent_label, options=sent_opts, index=0)
        change_label_style(sent_label, '20px', 'blue', 'Cambria')

    if opt == 6:  # Topic Modeling
        tm_label = "Choose a method of topic modeling:"
        tm_opts = ['Latent Dirichlet Allocation (LDA)', 
                   'Non-Negative Matrix Factorization (NMF)',
                   'BERTopic']
        tm_opt = st.sidebar.selectbox(tm_label, options=tm_opts)
        tm_opt = tm_opts.index(tm_opt)
        change_label_style(tm_label, '20px', 'blue', 'Cambria')

        topics_label = "Choose the number of topics to be generated:"
        topics_opts = list(np.arange(5, 21))
        topics_opt = st.sidebar.selectbox(topics_label, options=topics_opts, index=5)
        change_label_style(topics_label, '20px', 'blue', 'Cambria')
 
        scores_label = "Compute model evaluation scores?"
        scores_opts = ('No', 'Yes')
        scores_opt = st.sidebar.radio(scores_label, scores_opts, horizontal=True)
        scores_opt = scores_opts.index(scores_opt)
        change_label_style(scores_label, '20px', 'blue', 'Cambria')

    if opt == 7:  # Scripture Browsing
        lang_label = "English (WEB) / Chinese (CUV) Bible?"
        bi_opts = ('English', 'Chinese')
        bi_opt = st.sidebar.radio(lang_label, bi_opts, horizontal=True)
        bi_opt = bi_opts.index(bi_opt)
        change_label_style(lang_label, '20px', 'blue', 'Cambria')

        scop_label = "Choose OT / NT Book:"
        scop_opts = ['OT Book', 'NT Book']
        scop_opt = st.sidebar.selectbox(scop_label, options=scop_opts)
        scop_opt = scop_opts.index(scop_opt)
        change_label_style(scop_label, '20px', 'blue', 'Cambria')

        if scop_opt == 0:  # OT Book
            cat_label = "Choose an OT Book:"
            cat_opts = otbks
        else:  # NT Book
            cat_label = "Choose an NT Book:"
            cat_opts = ntbks

        cat_opt = st.sidebar.selectbox(cat_label, options=cat_opts,
                                       on_change=lambda: reset_label(ch_label))
        cat_opt = cat_opts.index(cat_opt)
        change_label_style(cat_label, '20px', 'blue', 'Cambria')

        bdf = util.extract(wdf, book=scop_opt*39+cat_opt+1)
        ch_label = "Choose a chapter:"
        ch_opts = list(bdf.chapter.unique())
        ch_opt = st.sidebar.selectbox(ch_label, options=ch_opts)
        ch_opt = ch_opts.index(ch_opt)
        change_label_style(ch_label, '20px', 'blue', 'Cambria')
 
    if opt > 1:
        show_label = "Show source code?"
        no_yes = ('No', 'Yes')
        show_code = st.sidebar.radio(show_label, no_yes, horizontal=True)
        show_code = no_yes.index(show_code)
        change_label_style(show_label, '20px', 'blue', 'Cambria')

    st.sidebar.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: blue;
            color: white;
        }
        div.stButton > button:hover {
            background-color: brown;
            }
       </style>""", unsafe_allow_html=True)

    #####
    # Process different options #
    #####
    if st.sidebar.button("Proceed"):
        if opt == 1:  # Show module code
            source(cwordtm.meta.get_submodule_info(mod_sel, 
                                                   detailed=bi_opt))
        elif opt == 2:  # Scripture Statistics
            df = wdf
            sdf = pivot.stat(df, code=show_code)
            sdf = sdf.rename(columns={'chapter': 'chapters',
                                      'verse': 'verses',
                                      'text': 'words'})
            bible_stat(sdf.reset_index(), util.bible_cat_info())

        elif opt == 3:  # Quotes from OT
            bible = web if bi_opt == 0 else cuv
            lang = 'en' if bi_opt == 0 else 'chi'
            df = wdf
            bk_ch = bdf.iloc[0].book + ' ' + str(ch_opt)
            ch_text = util.extract2(df, bk_ch)
            quot.show_quot(ch_text, lang=lang, threshold=thres_opt,
                           code=show_code)

        elif opt == 4:  # Wordcloud
            if ds_opt == 0:  # BBC News
                ndf = util.load_csv(bbc, doc_size=news_range, info=True)
                text_list = util.get_text_list(ndf, text_col='text')
                func = viz.show_wordcloud
                fig = func(text_list, bg='black', image=mask_opt, web_app=True, code=show_code)
                wordcloud(fig)

            elif ds_opt == 1:  # Bible
                bible = web if bi_opt == 0 else cuv
                if bible == web:
                    df = wdf
                else:
                    df = util.load_word(bible)
                if scop_opt == 0:  # Whole Bible
                    scdf = df
                elif scop_opt in [1, 2]:  # OT / NT
                    if cat_opt == 0:
                        scdf = df[df.testament==scop_opt-1]
                    else:  # Book
                        scdf = df[df.book_no==(scop_opt-1)*39+cat_opt]
                else:  # Category
                    scdf = df[df.cat_no==cat_opt]

                text_list = util.get_text_list(scdf)
                func = viz.show_wordcloud if bi_opt == 0 else viz.chi_wordcloud
                fig = func(text_list, bg='black', image=mask_opt, web_app=True, code=show_code)
                wordcloud(fig)

            else:  # Other Dataset
                if oth_opt != 0:
                    lim = 500  # Limit the no of source sentences
                    odf = util.load_csv(oth_file, doc_size=lim, info=True)
                    text_list = util.get_text_list(odf, text_col='text')
                    wc_func = viz.show_wordcloud if bi_opt == 0 else viz.chi_wordcloud
                    fig = wc_func(text_list, bg='black', image=mask_opt, web_app=True, code=show_code)
                    wordcloud(fig)
                else:
                    st.sidebar.write("No file selected!")

        elif opt == 5:  # Text Summary
            source_docs = None
            if ds_opt == 0:  # BBC News
                ndf = util.load_csv(bbc, doc_size=news_range, info=True)
                text_list = util.get_text_list(ndf, text_col='text')
                summary = ta.summary_en(text_list, sent_len=sent_opt, code=show_code)
                source_docs = ndf.iloc[news_range[0]-1:news_range[1]]

            elif ds_opt == 1:  # Bible
                bible = web if bi_opt == 0 else cuv
                if bible == web:
                   df = wdf
                else:
                    df = util.load_word(bible)
                if scop_opt == 0:  # Whole Bible
                    scdf = df
                elif scop_opt in [1, 2]:  # OT / NT
                    if cat_opt == 0:
                        scdf = df[df.testament==scop_opt-1]
                    else:  # Book
                        scdf = df[df.book_no==(scop_opt-1)*39+cat_opt]
                        if ch_opt > 0:
                            scdf = scdf[scdf.chapter==ch_opt]
                else:  # Category
                    scdf = df[df.cat_no==cat_opt]

                if bi_opt == 0:  # English
                    text_list = util.get_text_list(scdf)
                    summary = ta.summary_en(text_list, sent_len=sent_opt, code=show_code)
                else:  # Chinese
                    summary = ta.summary_chi(scdf, sent_len=sent_opt, code=show_code)

                source_docs = scdf.reset_index(drop=True)

            else:  # Other Dataset
                if oth_opt != 0:
                    lim = 500  # Limit the no of source sentences
                    odf = util.load_csv(oth_file, doc_size=lim, info=True)
                    if bi_opt == 0:  # English
                        text_list = util.get_text_list(odf, text_col='text')
                        summary = ta.summary_en(text_list, sent_len=sent_opt, code=show_code)
                    else:  # Chinese
                        summary = ta.summary_chi(odf, sent_len=sent_opt, code=show_code)
                    source_docs = odf.iloc[:lim]
                else:
                    st.sidebar.write("No file selected!")

            if ds_opt != 2 or oth_opt != 0:
                summary_div = ""
                for i, sent in enumerate(summary[:10]):
                    summary_div += "%02d) %s<br>" %(i+1, sent)
                summary_out(summary_div, source_docs)

        elif opt == 6:  # Topic Modeling
            tm_funcs = [tm.lda_process, tm.nmf_process, tm.btm_process]
            tm_func = tm_funcs[tm_opt]

            topics_opt = int(topics_opt)
            if ds_opt == 0:  # BBC News
                tmm = tm_func(bbc, num_topics=topics_opt, source=1,
                              text_col='text', doc_size=news_range,
                              eval=scores_opt, web_app=True,
                              timing=True, code=show_code)

            elif ds_opt == 1:  # Bible
                bible = web if bi_opt == 0 else cuv
                if bible == web:
                    df = wdf
                else:
                    df = util.load_word(bible)

                if scop_opt < 3:  # Whole Bible, OT, or NT
                    tm_cat = scop_opt
                else:  # Category
                    tm_cat = cat_short[cat_opt]

                chi_flag = False if bi_opt == 0 else True 
                tmm = tm_func(bible, num_topics=topics_opt,
                              text_col='text', cat=tm_cat, eval=scores_opt,
                              web_app=True, timing=True, code=show_code)

            else:  # Other Dataset
                if oth_opt != 0:
                    chi_flag = False if bi_opt == 0 else True 
                    lim = 500  # Limit the no of source sentences
                    tmm = tm_func(oth_file, num_topics=topics_opt, source=1,
                                  text_col='text', doc_size=lim, chi=chi_flag,
                                  eval=scores_opt, web_app=True, timing=True,
                                  code=show_code)
                else:
                    st.sidebar.write("No file selected!")

            if ds_opt < 2 or oth_opt != 0:
                if tm_opt == 0:
                    html_string = pyLDAvis.prepared_data_to_html(tmm.vis_data)
                    lda_viz(html_string)
                if tm_opt == 1:  # NMF
                    tm_viz("NMF-Topic Distribution", tmm.figures[0])
                if tm_opt == 2:  # BERTopic
                    btm_viz(tmm.figures)

        elif opt == 7:  # Scripture Browsing
            bible = web if bi_opt == 0 else cuv
            if bible == web:
                df = wdf
            else:
                df = util.load_word(bible)

            scdf = df[df.book_no==scop_opt*39+cat_opt+1]
            scdf = scdf[scdf.chapter==ch_opt+1]
            scols = scdf[['book', 'chapter', 'verse', 'text']]

            scripture_browsing(scols)

        else:
            st.sidebar.subheader("Invalid option!")

    if st.sidebar.button("Help"):
        help_page()

    st.sidebar.markdown('''<hr>''', unsafe_allow_html=True)
    st.sidebar.markdown('''<small>[Overview](https://github.com/drjohnnycheng/webapp/blob/main/README.md) | [CWordTM](https://github.com/drjohnnycheng/cwordtm) | March 2025</small>''', unsafe_allow_html=True)


if __name__ == "__main__":
    st.markdown("""
        <h1 style="font-family: 'Cambria';
                   font-size: 36px;
                   color: blue;">
        NLP Tasks with CWordTM (Version 0.6.0)
        </h1>
        """, unsafe_allow_html=True)

    output = st.empty()
    output.markdown("""
        <div style="font-family: 'Cambria';
                    font-size: 20px;
                    color: blue;">
        </div>
        """, unsafe_allow_html=True)

    with st_capture(output.code):
        main()
