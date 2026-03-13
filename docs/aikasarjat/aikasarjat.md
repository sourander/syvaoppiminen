---
priority: 800
---

# Aikasarjat

!!! warning

    Perinteisiä koneoppimismenetelmiä ja tilastollisia malleja käsittelevä osa tästä materiaalista tullaan todennäköisesti siirtämään jatkossa Johdatus koneoppimiseen -kurssille. Syy materiaalin tuomiseen aluksi tälle kurssille liittyy kurssien refaktorointijärjestykseen.

## Määritelmä

Aloitetaan määrittelemällä, mitä aikasarjat ovat. Yksinkertainen, yhden muuttujan aikasarja on helppo kuvata kuvaajalla, kuten alla:

```mermaid
xychart
    x-axis [jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec]
    y-axis "Myynti" 4000 --> 11000
    bar [5000, 6200, 4800, 6500, 7800, 6200, 8000, 9300, 7700, 9500, 10800, 9200]
    line [5000, 6200, 4800, 6500, 7800, 6200, 8000, 9300, 7700, 9500, 10800, 9200]

```

Tässä tapauksessa dataa on yhdeltä vuodelta ja se on kerätty kuukausittain – eli *granulariteetti* on kuukausittainen. Jokainen datapiste kuvaa tietyn ajanhetken (tässä tapauksessa kuukauden aloituspäivän) ja siihen liittyvän arvon (tässä tapauksessa myynti). Myyntidata on intuitiivinen esimerkki, mutta valtava osuus maailman aikasarjadatasta kertyy erilaisilta IoT-laitteilta. Aikasarja voi kuvastaa vaikkapa palvelinkeskuksen kiintolevyjen hajoamista: sarakkeista löytyy SMART-metriikoita ja binäärinen tieto siitä, onko levy hajonnut kyseisenä päivänä.

??? info "Taulukkomuoto"

    Taulukkomuodossa sama data näyttäisi tältä:

    | `month_start` | `sales` |
    | ------------- | ------- |
    | 2026-01-01    | 5000    |
    | 2026-02-01    | 6200    |
    | 2026-03-01    | 4800    |
    | 2026-04-01    | 6500    |
    | ...           | ...     |
    | 2026-11-01    | 10800   |
    | 2026-12-01    | 9200    |


### Komponentit

Kun puhutaan *ennustamisesta*, pyrimme mallintamaan aikasarjan tulevia arvoja. Kuvaajan tapauksessa tämä tarkoittaa, että meitä kiinnostaisi ensi vuoden tammikuu (helmikuu, maaliskuu, ...). Ennustamisen lisäksi on hyvä huomioida, että on olemassa myös aikasarjojen analysointia (engl. time series analysis), joka keskittyy ymmärtämään aikasarjan rakennetta, kuten kausivaihteluita, trendejä ja satunnaisuutta. Aikasarja-analyysiin kuuluu neljä pääkomponenttia [^ml-forecasting-py]:

* :one: **Trendi** eli pitkän ajan liike. Yllä olevassa kuvaajassa on noususuuntainen trendi.
* Lyhyen ajan vaihtelut:
    * :two: **Kausivaihtelut** eli säännölliset vaihtelut, jotka toistuvat tietyn ajan välein. Yllä olevassa kuvassa on 4 kuukauden "hain evä".
    * :three: **Sykliset vaihtelut** ovat nousuja ja laskuja, joiden ei ole kiinteää jaksoa (esim. suhdannevaihtelu).
* :four: **Satunnaisuus** eli täysin ennustamattomat pienet tai suuret vaihtelut. Tämä on kohinaa.

Nämä ovat tekstinä abstrakteja käsitteitä, joten kannattaa etsiä kuva avuksi. Voit löytyää kuvia esimerkiksi `statsmodel`-kirjaston esimerkistä [Seasonal-Trend decomposition using LOESS (STL)](https://www.statsmodels.org/stable/examples/notebooks/generated/stl_decomposition.html)

### Ennustamisen termistö

Datan suhteen muita tärkeitä termejä ovat [^ml-forecasting-py]:

* **(aika)granulariteetti**: Kuinka usein dataa kerätään (esim. päivittäin, kuukausittain, vuosittain).
* **aikahorisontti**: Kuinka pitkälle tulevaisuuteen haluamme ennustaa.
* **exogeeniset muuttujat**: Muut muuttujat (*engl. feature*), jotka vaikuttavat ennustettavaan arvoon, mutta eivät ole osa aikasarjaa. Esimerkiksi `is_holiday` tai `outdoor_temperature` tai `sensor_type`.
* **yksi- tai monimuuttuja**: Onko aikasarja yhden vai useamman muuttujan sarja (*engl. monovariate, multivariate*).
* **ennustehorisontin rakenne**: Ennustetaanko vain seuraava aika-askel (*engl. single-step*) vai useampi askel (*engl. multi-step*). Multi-step -ennusteita voidaan tehdä kolmella eri tavalla (plus yhdellä hybridillä, joka on pudotettu pois listalta):
    * **rekursiivinen ennuste** (*recursive multi-step*): Käytetään yhtä mallia, joka ennustaa seuraavan askeleen. Tulos syötetään takaisin malliin seuraavan askeleen ennustamiseksi.
    * **suora ennuste** (*direct multi-step*): Luodaan erillinen malli jokaiselle ennustettavalle aika-askeleelle (esim. yksi malli tunnille $t+1$ ja toinen tunnille $t+2$).
    * **moniulotteinen tuloste** (*multiple output*): Yksi malli ennustaa koko sekvenssin kerralla vektorina. Tämä on ==neuroverkoille luonnollisin tapa==.

Kannattaa tutustua näihin [skforecast.org: Intro to Forecasting](https://skforecast.org/0.20.1/introduction-forecasting/introduction-forecasting)-sivulla, jossa ne on esitelty visuaalisesti.

## Datan käsittely (Perinteinen)

!!! warning

    Tässä materiaalissa ei käsitellä puuttuvien arvojen imputointia syvällisesti. Ota kuitenkin huomioon, että aikasarjasta puuttuvat arvot täytyy käsitellä ennen mallinnusta esimerkiksi interpoloimalla, forward fill -menetelmällä tai poistamalla käyttökelvottomat rivit.

### LAG

Tässä kohtaa on hyvä erottaa kaksi perinnettä toisistaan. **Taulukkomuotoiset koneoppimismallit** – kuten lineaarinen regressio, random forest, XGBoost ja LightGBM – tarvitsevat yleensä eksplisiittisesti rakennetut viivepiirteet (*lag features*). Sen sijaan **klassiset aikasarjamallit**, kuten ARIMA tai Exponential Smoothing, mallintavat ajallista rakennetta suoremmin eivätkä välttämättä vaadi sitä, että rakennat lag-sarakkeet itse [^modern-ts-forecasting] [^ts-cookbook].

Jos käytössä on siis tavallinen regressiomalli tai boosting-malli, 1-ulotteinen aikasarja täytyy käytännössä kääntää moniulotteiseksi *sliding window* -menetelmällä. Tämän voi tehdä Pandasilla, SQL:llä, Excelillä tai valitsemallaan aikasarjakirjastolla (esim. `sktime` tai `MLForecast`). Koska SQL on yleismaailmallisesti ymmärrettävä kieli, näytetään esimerkki SQL:llä:

```sql
CREATE TABLE sales_features AS
SELECT
    month_start,                                       -- kuukauden aloituspvm
    LAG(sales, 2) OVER (ORDER BY month_start) AS lag_2, -- toissakuukauden myynti
    LAG(sales, 1) OVER (ORDER BY month_start) AS lag_1, -- viime kuukauden myynti
    sales                                              -- tämän kuukauden myynti
FROM sales;
```

| `month_start` | `lag_2 (x_0)` | `lag_1 (x_1)` | `sales (y)` |
| :-----------: | :-----------: | :-----------: | :---------: |
|  2026-01-01   | :down_arrow:  | :down_arrow:  |    5000     |
|  2026-02-01   | :down_arrow:  |     5000      |    6200     |
|  2026-03-01   |     5000      |     6200      |    4800     |
|  2026-04-01   |     6200      |     4800      |    6500     |
|      ...      |     4800      |     6500      |     ...     |
|      ...      |     6500      |      ...      |     ...     |

Lopputulos syntyy siten, että jokaiselle kuukaudelle luodaan uusi sarake (*engl. column*), jossa on kyseisen kuukauden myynti ja sitä edeltävien kuukausien myynnit. Näin saadaan luotua uusia ominaisuuksia (features), jotka kuvaavat aikasarjan rakennetta. Perinteinen tilastollinen malli käsittelee näitä samalla tavalla kuin muitakin ominaisuuksia. Jos kuvittelet tilalle lukemaan `n_rooms`, `distance_to_city_center` ja `area`, niin on helppo hyväksyä, että tilastollinen malli voi löytää painot, jotka kuvaavat, kuinka paljon kukin näistä ominaisuuksista vaikuttaa ennustettavaan arvoon. Mallille syötettäisiin siis:

```python
model.fit(
    X=df[["lag_2", "lag_1"]], 
    y=df["sales"]
)
```

??? tip "Mitä muuta voi lisätä?"

    ```sql
    CREATE TABLE sales_features AS
    SELECT
        month_start,
        sales,
        -- ....
        -- ...
        -- (endogeeninen) liukuva keskiarvo, joka kuvaa viimeisten 7 päivän myyntiä
        AVG(sales) OVER (
            ORDER BY month_start
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS rolling_7d_avg,
        -- kalenteripiirre: alkoiko kuukausi maanantaina?
        CASE 
            WHEN EXTRACT(ISODOW FROM month_start) = 1
            THEN 1 ELSE 0
        END AS month_started_on_monday
    FROM sales
    ```

    Taulusta pitäisi lopuksi poistaa tai imputoida rivit, joissa on `NULL`-arvoja, koska `LAG()` tuottaa niitä ensimmäisille riveille.

### Kalenteri- ja ulkoiset piirteet

Pelkkä viivehistoria ei riitä kaikissa ongelmissa. Monissa liiketoiminta- ja IoT-ongelmissa ennuste tarkentuu merkittävästi, kun mukaan lisätään:

* **kalenteripiirteitä**, kuten viikonpäivä, kuukausi, tunti, lomapäivä tai kampanjapäivä
* **exogeenisiä muuttujia**, kuten lämpötila, hinta, säätila tai sensorin tyyppi
* **tilastollisia aggregaatteja**, kuten liukuvia keskiarvoja, minimejä, maksimeja ja volatiliteettia

Tärkeä käytännön sääntö on tämä: piirteessä saa käyttää vain sellaista tietoa, joka olisi ollut aidosti saatavilla ennustushetkellä. Jos rakennat esimerkiksi 7 päivän liukuvan keskiarvon, sitä ei saa laskea tulevien päivien arvoista. Muuten syntyy *data leakage*.

### Train-test-jako

Tavallisessa koneoppimisessa aineisto voidaan usein sekoittaa ja jakaa satunnaisesti opetus- ja testijoukkoihin. Aikasarjojen kanssa näin ei voida tehdä, sillä ajan suuntaa on ehdottomasti kunnioitettava: tarkoituksena on ennustaa tulevaisuutta menneisyyden datan perusteella, eikä päinvastoin [^dlwithpython]. Siksi testijoukoksi on aina varattava aikasarjan kronologisesti tuorein osa [^ts-cookbook]. Sama periaate pätee myös mallin validointiin. Aikasarjojen ristiinvalidoinnissa (cross-validation) ei voida käyttää perinteistä k-fold-menetelmää datan satunnaistuksella. Sen sijaan hyödynnetään aikasidonnaisia menetelmiä, kuten laajenevaa ikkunaa (expanding window) tai liukuvaa ikkunaa (sliding window), jotta mallin arviointi tapahtuu aina aidosti tulevaa ennustamalla [^modern-ts-forecasting].

Käytännössä yksinkertainen jako voi näyttää tältä:

* vanhin 70–80 % datasta koulutukseen
* seuraava siivu validointiin
* tuorein siivu testiin

Jos dataa on vähän, erillisen validointijoukon sijasta voidaan käyttää useita peräkkäisiä validointi-ikkunoita.

### Stationaarisuus

Stationaarisuus on perusoletus klassisille autoregressiivisille malleille, kuten ARIMA:lle. Stationaarinen aikasarja tarkoittaa sitä, että sen tilastolliset perusominaisuudet (mean, std) pysyvät ajan suhteen vakaina. Riippuvuus menneisyyden arvoihin määräytyy ensisijaisesti viiveen, ei absoluuttisen ajanhetken, perusteella [^modern-ts-forecasting].

Ongelmana on, että suuri osa todellisen maailman datasta sisältää trendiä, kausivaihtelua tai rakennemuutoksia, jolloin sarja ei ole luonnostaan stationaarinen [^ts-cookbook]. Tällöin datasta tehdään väkisin stationaarista metodeilla, kuten:

* **differointi** (*differencing*) trendin poistamiseen (eli $y_t' = y_t - y_{t-1}$)
* **kausidifferointi** kausirakenteen poistamiseen (eli $y_t'' = y_t - y_{t-m}$, missä $m$ on kausijakson pituus)
* **logaritmimuunnos** tai muuta skaalaavaa muunnosta varianssin vakiointiin

Kaikki perinteiset mallit eivät kuitenkaan vaadi stationaarisuutta. Esimerkiksi puupohjaiset regressiomallit voivat toimia epästationaarisella datalla. [^ml-forecasting-py]

### Autokorrelaatio ja piirteiden valinta

Autokorrelaatio kertoo, kuinka vahvasti sarja korreloi oman menneisyytensä kanssa. Tämä on sinulle kielimalleista jo tuttua: edelliset sanat korreloivat siihen, mikä tulee seuraavaksi. Jos tämän päivän myynti muistuttaa eilisen myyntiä, yhden päivän viiveellä on positiivinen autokorrelaatio. Aikasarja-analyysissä tätä tarkastellaan usein kahdella työkalulla [^ml-forecasting-py] [^ts-cookbook]:

* **ACF** (*autocorrelation function*): näyttää, kuinka vahva korrelaatio kullakin viiveellä on
* **PACF** (*partial autocorrelation function*): näyttää viiveen "oman" vaikutuksen, kun välistä tulevien lyhyempien viiveiden vaikutus on vakioitu

Idea on seuraava:

* jos ACF:ssä näkyy piikkejä viiveillä 7, 14 ja 21, datassa voi olla viikkokausivaihtelua
* jos PACF:ssä näkyy selvä piikki viiveellä 1 tai 2, nämä lagit voivat olla hyödyllisiä autoregressiivisessä mallissa
* jos ACF hiipuu hitaasti, sarjassa voi olla trendiä eikä se ehkä ole stationaarinen

Näihin kannattaa tutustua kuvien kautta. Myös tähän löytyy `statsmodels`-kirjaston esimerkeistä valmis kuvaaja, esimerkiksi: [Autoregressive Moving Average (ARMA): Sunspots data](https://www.statsmodels.org/stable/examples/notebooks/generated/tsa_arma_0.html)

Taulukkomuotoisten mallien kanssa ACF ja PACF auttavat ennen kaikkea valitsemaan, mitkä viiveet kannattaa ottaa mukaan piirteinä. Jos sarjassa on vahva 24 tunnin tai 7 päivän rytmi, nämä viiveet kannattaa usein mallintaa eksplisiittisesti.

### Lokaalit mallit

**Lokaali malli** tarkoittaa sitä, että jokaiselle aikasarjalle koulutetaan oma malli. Jos ennustat 500 tuotteen kysyntää, lokaalissa lähestymistavassa rakennetaan 500 erillistä mallia [^modern-ts-forecasting]. Tämä on klassisen aikasarja-analyysin oletustapa.

Lokaalien mallien etuja ovat:

* helppo tulkittavuus yhdelle sarjalle kerrallaan
* hyvä toiminta silloin, kun sarjoja on vähän mutta kutakin sarjaa on mitattu pitkään
* yksinkertainen ajatusmalli: yhden sensorin historiaa käytetään saman sensorin tulevaisuuden ennustamiseen

Haittoja puolestaan ovat:

* mallit eivät jaa oppimaansa keskenään
* ylläpidettävien mallien määrä kasvaa nopeasti
* lyhyiden tai kohinaisten sarjojen ennustaminen on vaikeaa, koska yhdellä sarjalla on vähän opetusdataa

Lokaali malli on usein erinomainen valinta silloin, kun ennustettava ilmiö on yksi selkeä sarja – esimerkiksi sähkönkulutus yhdessä rakennuksessa tai yhden tuotteen kysyntä yhdessä myymälässä.

### Sama videona

Alla olevasta upotetusta videosta löydät samat asiat kuin yltä (ja vähän enemmänkin) selkeän visuaalisessa muodossa.

<iframe width="560" height="315" src="https://www.youtube.com/embed/9QtL7m3YS9I?si=-t5pqekYzaRJjMXf" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

**Video:** *Kishan Manani esittelee PyData London 2022 -konferenssissa aihetta "Feature Engineering for Time Series Forecasting".*


## Datan käsittely (Neuroverkot)

### LAG

Perinteisissä regressiomalleissa viivemuuttujien rakentaminen sarakkeiksi on yleensä välttämätöntä. Neuroverkoissa ajallinen historia syötetään sen sijaan useimmiten **sekvenssinä**, ei yksittäisinä lag-sarakkeina. Tämä ei tarkoita, että historiaikkuna katoaisi kokonaan – se vain esitetään eri muodossa [^ml-forecasting-py].

Esimerkiksi 48 tunnin historia voidaan antaa mallille yhtenä syötesekvenssinä, jolloin RNN-, LSTM- tai Transformer-malli saa itse oppia, mihin kohtiin historiassa kannattaa kiinnittää huomiota. Käsin rakennettuja `lag_1`, `lag_2`, `lag_24` -sarakkeita ei siis välttämättä tarvitse materialisoida taulukoksi, vaikka sama informaatio onkin yhä mukana.

Käytännössä neuroverkolle syötettävä data näyttää rakenteeltaan tältä:

```
Input shape: (batch_size, sequence_length, features)
Output shape: (batch_size, forecast_horizon[, target_features])
```

Tässä `sequence_length` määrittää sen aikaikkunan, jonka verkko "näkee" kerralla. Jos ennustetaan vain yhtä kohdemuuttujaa, ulostulo voi olla 2-ulotteinen. Monimuuttujaisessa ennusteessa myös tulosteeseen jää oma piirreulottuvuus.

### Skaalaus

Vaikka neuroverkot eivät vaadi stationaarisuutta kuten ARIMA (vaikka kenties voivatkin siitä hyötyä?), ne hyötyvät lähes aina siitä, että data on skaalattu järkevästi. Tämän pitäisi olla kertausta. Käytännössä tämä tarkoittaa:

* standardointia ($z$-score) tai min-max-skaalausta
* log-muunnosta voimakkaasti vinoille (positiivisille) sarjoille

Skaalaus pitää aina sovittaa vain koulutusdataan ja käyttää sitten samoja parametreja validointi- ja testidataan. Muuten taas vuotaa tulevaisuuden tietoa väärään paikkaan. 

### Train-test-jako

Neuroverkojen suhteen train-test-jako noudattaa samaa periaatetta kuin perinteisissä malleissa, mutta toteutus näyttää usein hieman erilaiselta. Yksittäiset opetusnäytteet eivät ole irrallisia rivejä vaan **ikkunoita**: menneisyysikkuna syötetään sisään ja tulevaisuusikkuna annetaan tavoitteeksi. Toivon mukaan tunnistat, että tämä ei sinänsä poikkea millään tavoin mistään kurssin aiemmista neuroverkoista käytäntönä.

### Stationaarisuus

Kuten todettua, stationaarisuuden saavuttaminen on perinteisissä menetelmissä usein työläs esiaskel. Syväoppimisessa tämä taakka kevenee huomattavasti:

> "Neural networks can be useful for time series forecasting problems by eliminating the immediate need for massive feature engineering processes, data scaling procedures, and making the data stationary by differencing."
>
> — Lazzeri [^ml-forecasting-py]

Neuroverkot kykenevät oppimaan monimutkaisia epälineaarisia riippuvuuksia suoraan kohinaisesta ja "epätäydellisestä" datasta. Ne pystyvät käsittelemään datan sisäisiä rakenteita, kuten trendejä ja kausivaihteluita, ilman että niitä täytyy välttämättä poistaa etukäteen matemaattisilla muunnoksilla [^ml-forecasting-py].

Tämä ei kuitenkaan tarkoita, että kaikki esikäsittely olisi turhaa. Käytännössä seuraavat toimet ovat usein edelleen hyödyllisiä:

* poikkeavien arvojen tunnistaminen
* skaalauksen tekeminen
* voimakkaasti vinojen jakaumien muuntaminen
* kausikalenterin ja exogeenisten piirteiden lisääminen

Hyvä nyrkkisääntö on siis: neuroverkot vähentävät käsin tehtävän aikasarja-analyysin määrää, mutta eivät poista tarvetta ymmärtää dataa.

### Autokorrelaatio ja piirteiden valinta

Neuroverkkojen yhteydessä ACF- ja PACF-kuvaajia ei yleensä käytetä yhtä suoraviivaisesti mallin rakenteen valintaan kuin ARIMA-maailmassa. Verkko oppii painot sekvenssistä itse, eikä analyytikon tarvitse erikseen päättää jokaista yksittäistä lagia.

Tästä huolimatta autokorrelaation ymmärtäminen on edelleen hyödyllistä, koska sen avulla voi:

* valita järkevän `sequence_length`-arvon
* havaita mahdollisen kausijakson, kuten 24 tuntia tai 7 päivää
* arvioida, onko lyhyen vai pitkän muistin arkkitehtuuri todennäköisesti hyödyllinen
* rakentaa vahvoja baseline-malleja vertailua varten

Jos esimerkiksi ACF näyttää selviä piikkejä vuorokausi- ja viikkoviiveillä, olisi outoa syöttää mallille vain neljän tunnin historia ja odottaa sen oppivan koko ongelman. Datan perusrakenne kannattaa siis ymmärtää, vaikka varsinainen piirteiden poiminta jätetäänkin verkolle.

### Globaalit mallit

**Globaali malli**, johon viittaa myös termi *cross-learning*, tarkoittaa sitä, että yksi ja sama malli koulutetaan usean eri aikasarjan yli. Tällöin malli oppii yhteisiä rakenteita sarjojen välillä: esimerkiksi sen, että viikonloput pudottavat kysyntää, tai että kampanjat kasvattavat myyntiä useissa tuoteryhmissä samaan tapaan [^deep-ar] [^smyl] [^modern-ts-forecasting]. DeepAR esitteli tämän termin maailmalle. [^deep-ar]

Tämä ajatus on modernin koneoppimisen tarjoama merkittävä etu. Sen sijaan että koulutetaan 500 erillistä mallia 500 tuotteelle, voidaan kouluttaa yksi globaali malli, joka näkee kaikkien tuotteiden historian. Sarjakohtaiset erot voidaan syöttää mallille esimerkiksi tunnisteella `series_id`, `tuoteryhmä` tai `myymälä`. 

> "The strength of ML algorithms, and in fact the requirement for their successful use, is cross-learning, i.e., using many series to train a single model. This is unlike standard statistical time series algorithms, where a separate model is developed for each series."
>
> — Slawek Smyl [^smyl]

## Mallin koulutus (Perinteinen)

Perinteisten mallien vahvuus on siinä, että ne ovat usein nopeita, tulkittavia ja kilpailukykyisiä yllättävän vahvoina baselineina. Käytännössä on tavallista kokeilla useita eri malliperheitä rinnakkain ja valita niistä paras validaation perusteella [^modern-ts-forecasting]. Huomaa, että kaikkea mallinnusta ei tarvitse tehdä ilman apuja. Esimerkiksi Nixtlan `StatsForecast` ja `MLForecast` voivat olla avuksi. Pienten datasettien kanssa, erityisesti asiaa opiskellessa, myös `skforecast` on hyvä ja helppokäyttöinen työkalu.

> "Time series forecasting has been around since the early 1920s, and through the years, many brilliant people have come up with different models, some statistical and some heuristic-based. I refer to them collectively as classical statistical models or econometrics models, although they are not strictly statistical/econometric."
>
> — Joseph & Tackes [^modern-ts-forecasting]

### Aloita baselinesta

Ennen kuin koulutat yhtään monimutkaista mallia, rakenna vähintään yksi yksinkertainen baseline. Aikasarjoissa hyvä baseline ei ole satunnainen arvaus vaan jokin ilmiön rakennetta hyödyntävä nyrkkisääntö. Tyypillisiä baselineja ovat:

* **naive**: huominen arvo = tämän päivän arvo (eli $y_{t+1} = y_t$)
* **seasonal naive**: ensi maanantai = viime maanantai (eli $y_{t+7} = y_t$)
* **moving average**: ennuste = viimeisten havaintojen keskiarvo

Huomaa, että nämä ==eivät siis varsinaisesti ennusta mitään==. Jos valitsemasi *forecast horizon* on 7 päivää, ja ajat ennusteen sunnuntaina, niin naive baseline toimii siten, että:

* maanantain arvo on sunnuntain arvo
* tiistain arvo on sunnuntain arvo
* keskiviikon arvo on sunnuntain arvo
* ...
* joka päivä jatkossa on sama kuin sunnuntai

![](../images/800_gemini-naive-classifier-SAME-stamp.jpg)

**Kuva 1:** *Nano Banana 2:n näkemys Naive Baseline -hahmosta, joka ennustaa kalenteria aina samalla kumileimasimella.*

!!! warning

    Jos monimutkainen malli ei voita näitä, ongelma on joko mallissa, datassa tai arviointitavassa. Mikäli törmäät online-kirjoitukseen, jossa rakennetaan monimutkainen malli ennustamaan esimerkiksi osakehintaa vertaamatta sitä edes naive-baselineen, voit olla melko varma siitä, että kyseessä on clickbait-artikkeli.

### ARIMA ja SARIMA

ARIMA rakentuu kolmesta osasta: autoregressiivinen osa (AR), differointi (I) ja liukuva keskiarvo virheille (MA). Mallin hyperparametrit kirjoitetaan yleensä muotoon $(p, d, q)$, ja kausillinen versio laajennetaan usein muotoon $(P, D, Q, m)$ [^ts-cookbook]. ARIMA-tyyppiset mallit sopivat erityisesti silloin, kun sarjoja on vähän, mieluiten yksi, ja ongelma on melko "siisti", taipuen helposti stationaariseksi.

Kyseessä on erittäin klassinen lokaali malli. Se on usein hyvä ensimmäinen vakava vertailukohta yhdelle selkeälle aikasarjalle.

### Exponential Smoothing ja ETS

Exponential Smoothing -malliperhe painottaa tuoreita havaintoja vanhoja enemmän. ETS-mallit (*Error, Trend, Seasonality*) ovat erityisen käyttökelpoisia silloin, kun sarjassa näkyy selkeä taso-, trendi- ja kausirakenne [^modern-ts-forecasting]. Käytännössä tämä on vaihtoehto ARIMA:lle, ja on myös *state space model*, joten se on helppo napata vertailuun mukaan.

### Lineaarinen regressio viivepiirteillä

Kun aikasarja muutetaan taulukkomuotoon lag-piirteiden avulla, tavallinen lineaarinen regressio muuttuu täysin käyttökelpoiseksi ennustajaksi. Tällöin malli oppii painot esimerkiksi piirteille `lag_1`, `lag_24`, `lag_168`, `is_weekend` ja `temperature`. [^modern-ts-forecasting] Jos piirteitä on paljon, käytetään usein regularisoituja versioita, kuten Ridge- tai Lasso-regressiota.

!!! tip "Regressio ja forecast horizon"

    Huomaa, että lineaarinen regressio on luonteeltaan single-step-malli. Jos haluat ennustaa useamman askeleen päähän, sinun täytyy joko:

    * käyttää rekursiivista lähestymistapaa, jossa ennuste syötetään takaisin malliin seuraavan askeleen ennustamiseksi
    * tai rakentaa erillinen malli jokaiselle ennustettavalle askeleelle

### LightGBM ja muut boosting-mallit

Kun lagit, kalenteripiirteet ja exogeeniset muuttujat on rakennettu taulukoksi, puupohjaiset boosting-mallit ovat usein erittäin vahvoja ennustajia. Ne pystyvät mallintamaan epälineaarisuuksia ja piirteiden välisiä interaktioita ilman, että analyytikon tarvitsee määritellä niitä käsin.

LightGBM, XGBoost tai Catboost ovat käytännössä hyvä valinta silloin, kun ilmiössä on epälineaarisuutta.

Aiheeseen voit tutustua esimerkiksi sk-forecastin examples-osion artikkelia lukemalla: [Forecasting time series with gradient boosting: Skforecast, XGBoost, LightGBM, Scikit-learn and CatBoost](https://cienciadedatos.net/documentos/py39-forecasting-time-series-with-skforecast-xgboost-lightgbm-catboost.html).

### Validointi ja mallin valinta

Perinteisten mallien koulutusprosessissa ei riitä, että ajetaan yksi malli yhdellä parametrilla. Tämä on toivon mukaan Johdatus koneoppimiseen -kurssilta tuttua. Mallit ovat kevyitä kouluttaa, ja frameworkit tarjoavat helpon tavan kokeilla useita malleja. Tyypillinen työnkulku on:

1. rakenna naiivi baseline
2. valitse useita malleja ja malliperheitä
3. kouluta kaikki mallit koulutusdatalla
4. arvioi kaikki mallit validointidatalla ja vertaa niitä baselineen
5. valitse paras malli ja arvioi se testidatalla
6. (*sneak peak seuraavasta otsikosta*) tulos huono ja paljon dataa? Nyt voisi olla oikea aika kaivaa LSTM esiin.

Tyypillisiä arviointimittareita ovat MAE, RMSE, MAPE ja sMAPE. Miten valita näiden väliltä, ja mitä backtesting edes ylipäätänsä on? Kannattaa katsoa seuraava video:

<iframe width="560" height="315" src="https://www.youtube.com/embed/dSTXd8Hx728?si=OSn2ul0KDPBpK4rN" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

**Video:** *Kishan Manani esittelee PyData London 2024 -konferenssissa aihetta "Backtesting and error metrics for modern time series forecasting". Huomaa, että Manani ei maalaa pelkän ruusuista kuvaa neuroverkoista videossa.*

!!! tip "State ja backtesting"

    State space -mallit, kuten ARIMA, käsittelevät aikaa hyvin eri tavalla kuin lag-piirteisiin perustuvat tilattomat (stateless) regressiomallit. Ne ylläpitävät sisäistä tilaa, joka riippuu havaintojen järjestyksestä. Tämä tarkoittaa, että ARIMA-malli voi tuottaa ennusteita vain viimeisestä havainnosta eteenpäin.

    > "Unlike machine learning models, statistical models like ARIMA maintain an internal state that depends on the sequence of observations. They can only generate predictions starting from the last observed time step — they cannot "jump" to an arbitrary point in the future without knowing all previous values. During backtesting, when the validation window moves forward, the model must be refitted to incorporate the new observations and update its internal state."
    >
    > — Rodrigo, Ortiz and Akay [^skforecast-statistical]

!!! tip

    Saatat törmätä myös termeihin *in-sample* ja *out-of-sample*. In-sample-ennuste tarkoittaa mallin kykyä selittää dataa, jonka se on nähnyt koulutuksen aikana. Forecasting-tehtävässä kiinnostavampi suure on lähes aina *out-of-sample*-suorituskyky eli ennuste täysin tulevaan aikaan [^modern-ts-forecasting].


## Mallin koulutus (Neuroverkot)

Neuroverkot tulevat kuvaan mukaan erityisesti silloin, kun dataa on paljon, sarjoja on paljon tai ongelma on niin monimutkainen, että käsin rakennetut lag-piirteet alkavat tuntua keinotekoisilta. Tämä ei silti poista sitä vanhaa sääntöä, että baseline pitää rakentaa ensin. [^ml-forecasting-py] Huomaa, että vaikka tällä kurssilla saatetaan harjoitella aikasarjaennustusta PyTorchilla ilman aikasarjaan erikoistunutta frameworkia, *käytännön projektissa* halunnet ottaa esimerkiksi Nixtlan `NeuralForecast`-kirjaston avuksi.

> "Generally, deep learning methods have been developed and applied to univariate time series forecasting scenarios (...) For this reason, they have often performed worse than naïve and classical forecasting methods, such as autoregressive integrated moving average (ARIMA). This has led to a general misconception that deep learning models are inefficient in time series forecasting scenarios."
>
> — Lazzeri [^ml-forecasting-py]

### Baseline edelleen ensin

Neuroverkkoa ei pidä arvioida tyhjiössä. Vertailukohtana pitäisi olla vähintään (seasonal) naive ja mielellään myös yksi vahva perinteinen malli, kuten LightGBM. Muuten on mahdotonta sanoa, toiko raskas arkkitehtuuri oikeasti lisäarvoa.

### RNN-perhe

RNN-perhe on luonnollinen valinta, kun ongelmassa on merkittäviä ajallisia riippuvuuksia. Aivan kuten lauseiden kanssa, LSTM ja GRU mallit käsittelevät pitkän ajan riippuvuuksia vahvemmin kuin vanilla RNN.

Näiden mallien koulutuksessa valitaan vähintään:

* historiaikkunan pituus (`sequence_length`)
* ennustehorisontti
    * single-step (many-to-one)
    * multi-step (many-to-many)
* tappiofunktio, kuten MAE, MSE tai Huber

### Transformer-pohjaiset mallit

Transformer-arkkitehtuurit ovat nousseet myös aikasarjaennusteisiin, erityisesti silloin kun sekvenssit ovat pitkiä, exogeenisiä piirteitä on paljon tai aikariippuvuudet ovat monimutkaisia. Aiemmin oppimasi self-attention-mekanismi mahdollistaa sen, että malli voi tarkastella koko historiaa kerralla ja oppia, mitkä osat historiasta ovat relevantteja ennusteen kannalta.

> "Transformers have gained popularity for time series modeling due to their ability to capture long-sequence interactions more effectively than RNN models (...) Many real-world applications, such as weather forecasting, traffic prediction, industrial process controls, and electricity consumption planning, require the prediction of long sequence time series."
>
> — Nicole Koenigstein [^transformers-def-guide]

Jos haluat tutustua yhdenlaiseen tiettyyn toteutukseen, katso esimerkiksi Woo et. al. malli nimeltään Moirai: [ArXiV: Unified Training of Universal Time Series Forecasting Transformers](https://arxiv.org/abs/2402.02592)

### Tappiofunktio ja ennustetapa

Neuroverkkojen koulutuksessa on päätettävä myös, mitä tarkalleen ennustetaan. Kurssin kannalta on hyvä huomata, että juuri **multiple output** sopii neuroverkoille hyvin, koska verkko voi tuottaa koko horisontin yhdellä ulostulokerroksella. Jos teet regressiota, kurssilta jo aiemmin tuttu MSE on oiva valinta tappiofunktioksi. [^dl-for-ts-cookbook]

### Koulutuslooppi käytännössä

Varsinainen koulutus ei poikkea mistään, mitä et olisi jo kurssin aikana nähnyt. Syötetään dataa ikkunamuodossa, lasketaan tappio, backpropagoidaan ja päivitetään painot. Ainoa ero on se, että data on järjestetty sekvensseiksi eikä yksittäisiksi riveiksi. Aivan kuten lauseita kouluttaessa, myös aikasarjojen kanssa on tärkeää pitää huolta siitä, että `y` sisältää `forecast_horizon`-määrän verran tulevaisuuden arvoja, ja `x_sample` sisältää `sequence_length`-määrän verran menneisyyden arvoja. Aivan kuten lauseiden kanssa, aikasarjojen kanssa nämä yksittäiset ikkunat voi syöttää malliin missä tahansa järjestyksessä, mutta train-test-jako ja backtesting on tehtävä siten, että mallin ei koskaan anneta nähdä tulevaisuuden dataa.

## Tehtävät

!!! question "Tehtävä: Metro Interstate Traffic"

    Aja `800_metro_interstate_traffic.py`-notebook sekä sinällään että muokattuna. Muokkaus, mikä sinun tulee tehdä, on vaihtaa skaalausmenetelmä `StandardScaler`ista `MinMaxScaler`iin. Tarkkaile, miten se vaikuttaa suorituskykymittareihin. Tarkalleen ottaen sinun tulee muokata koodia muutamasta paikasta:

    1. Lisää import
    2. Vaihda $y$ skaalaus. $X$:ään ajettava Pipeline saa pysyä ennallaan.
    3. Rajoita mallin output positiivisiin lukuihin. (Vinkki: ReLU)
    4. Muokkaa TensorBoardiin tallentuvan ajon nimeä, jotta tunnistat eri ajot.

    Jos haluat haastaa itseäsi, parametrisoi tämä siten, että voit vaihtaa skaalausmenetelmää yhden hyperparametrin avulla. Kun olet valmis, kokeile rohkeasti muokata myös muita hyperparametreja.


## Lähteet

[^ml-forecasting-py]: Lazzeri, F. *Machine Learning for Time Series Forecasting with Python*. 2020. Wiley.
[^modern-ts-forecasting]: Joseph, M. & Tackes, J. *Modern Time Series Forecasting with Python - Second Edition*. Packt. 2024.
[^ts-cookbook]: Atwan, T. *Time Series Analysis with Python Cookbook - Second Edition*. Packt. 2026.
[^dlwithpython]: Watson, M & Chollet, F. *Deep Learning with Python, Third Edition*. Manning. 2025.
[^deep-ar]: Salinas, D., Flunkert, V. & Gasthaus, J. "DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks". 2017. https://arxiv.org/abs/1704.04110
[^smyl]: Smyl, S. "A hybrid method of exponential smoothing and recurrent neural networks for time series forecasting". 2020. https://www.sciencedirect.com/science/article/abs/pii/S0169207019301153
[^skforecast-statistical]: Rodrigo, J. Ortiz, J. & Akay, R. *Forecasting with statistical models*. 2026. https://cienciadedatos.net/documentos/py77-forecasting-statistical-models.html
[^transformers-def-guide]: Koenigstein, N. *Transformers: The Definitive Guide*. O'Reilly. 2026.
[^dl-for-ts-cookbook]: Cerqueira, V. & Roque, L. *Deep Learning for Time Series Cookbook*. Packt. 2024.
