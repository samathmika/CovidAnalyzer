from flask import Flask ,render_template
from flask_fontawesome import FontAwesome
import stats
import titletophome 
import tweetstop
#import CovidAnalyzer
import sentitop
app = Flask(__name__)
fa = FontAwesome(app)


@app.route('/home')
def home():
    
    return render_template("index.html",tottweet=titletophome.tottweet,engtweet=titletophome.engtweet,pos=sentitop.pnn[0],info=titletophome.info,ver=titletophome.ver)
@app.route('/positive')
def positive():
    
    return render_template("index2.html",var1=stats.favb,var2=stats.favbfols,var3=stats.folbfavs)
 

@app.route('/tweets')
def tweets():
    return render_template("index3.html",info0=tweetstop.info[0],info1=tweetstop.info[1],info2=tweetstop.info[2],sad0=tweetstop.sad[0],sad1=tweetstop.sad[1],sad2=tweetstop.sad[2],angry0=tweetstop.angry[0],angry1=tweetstop.angry[1],angry2=tweetstop.angry[2],happy0=tweetstop.happy[0],happy1=tweetstop.happy[1],happy2=tweetstop.happy[2])

@app.route('/sentiment')
def sentiment():
    return render_template("index4.html",post=sentitop.pnn[0],negt=sentitop.pnn[1],neut=sentitop.pnn[2],posp=sentitop.posp,neup=sentitop.neup,negp=sentitop.negp)

@app.route('/data_visualization')
def data_visualisation():
    return render_template("chartjs.html",date=titletophome.date )
@app.route('/weekly_updates')
def weekly_updates():
    return render_template("a.html")
if __name__ =='__main__':
    app.run()

