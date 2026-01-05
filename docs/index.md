# Syväoppiminen I

!!! warning

    Tämä sivusto on kokonaisuudessaan työn alla. Valmistuu keväällä 2026.

Tervetuloa kurssille! Tämä kurssi on jatkoa Johdatus koneoppimiseen -kursille. Kurssilla syvennytään koneoppimisen keskeiseen osa-alueeseen, syväoppimiseen (deep learning). Aiheita käydään läpi lyhyesti teoreettisesta näkökulmasta, mutta pääpaino on käytännön toteutuksissa. Viralliset oppimistavoitteet löydät OPS:sta, mutta pääpiirteittäin kurssin jälkeen:

* Osaat selittää syväoppimisen keskeiset käsitteet 5-vuotiaalle.
* Olet kouluttanut esimerkkien mukaiset mallit.
* ... joiden pohjalta olet luonut omia malleja.
* Olet luonut oppimispäiväkirjan, joka mahdollistaa kertauksen ja jatko-opiskelun.


Tutustut kurssin aikana kirjoihin [Understanding Machine Learning](https://udlbook.github.io/udlbook/) ja [Hands-On Machine Learning with Scikit-Learn and PyTorch](https://learning.oreilly.com/library/view/hands-on-machine-learning/9798341607972/). Kurssin jälkeen sinulla on valmiudet käyttää syväoppimista projektikursseilla ja työelämässä.

## Tehtävät

Kurssi sisältää tehtäviä, jotka on tarkoitettu tehtäviksi kurssin aikana. Tehtävät löytyvät kunkin osion lopusta. Lisäksi **kaikki** tehtävät ovat koostettuna [Tehtäväkooste](exercises.md)-sivulle. Tehtävät palautetaan [Oppimispäiväkirja 101](https://sourander.github.io/oat/) -ohjeistuksen mukaisesti eli Gitlab Pages:ssa hostattuna staattisena sivustona.

Useat tehtävät viittaavat [Marimo](https://marimo.io/)-työkalulla tehtyihin notebookeihin. Kyseessä on Jupyter Notebook -työkalun seuraaja. Löydät notebookit [gh:sourander/syvaopiminen/notebooks](https://github.com/sourander/syvaoppiminen/tree/main/notebooks) -polusta *tämän kurssin* repositoriosta. Notebookien käyttö neuvotaan kurssivideoilla ja tukisessioissa.

## Numerointi

Kurssiaiheet ovat numeroitu sataluvuilla. Otetaan esimerkiksi kuvitteelliset luvut 1 ja 2:

```
1. Eläinkunta:
    (100:) Nisäkkäät
    (110:) Linnut
2. Ohjelmointikielet:
    (200:) Python
    (210:) Rust
```

Aiheeseen **Eläinkunta** liittyvät aineistot tunnistat sataluvulla `1xx`, ja aiheeseen **Ohjelmointikielet** liittyvät aineistot tunnistat satakymmenluvulla `2xx`. Esimerkiksi

* `notebooks/nb/100/`
    * `100_marsu.py` 
    * `101_leijona.py`
    * `102_kissa.py`
    * `110_varis.py` (seuraava kymppi eli linnut)
    * `111_kotka.py`
* `notebooks/nb/200/`
    * `200_python_alkeet.py`
    * `210_rust_alkeet.py` (seuraava kymppi eli Rust)

Sama pätee esimerkiksi kurssin videoihin. Jos videon otsikossa on luku välillä `100-199`, tiedät, että se liittyy aiheeseen Eläinkunta. Jos taas videon otsikossa on luku välillä `200-299`, tiedät, että se liittyy aiheeseen Ohjelmointikielet. Kymmenluvusta tunnistat tarkemman aiheen.

## Koodin ajaminen

Kurssilla ei pelkästään katsella koodia vaan sitä ajetaan myös. Voit ajaa koodit seuraavilla tavoilla, järjestettynä suositelluimmasta vähiten suositeltuun:

* Paikallisesti `uv`:lla
* Disco Coder -ympäristössä
* Jupyter Hubissa
* Google Colabissa

Vaihtoehtoja esitellään kurssivideoilla ja tukisessioissa. Mikäli sinulla on oma näytönohjain, on äärimmäisen suositeltavaa opetella ajamaan koodia paikallisesti kyseistä näytönohjainta hyödyntäen. AI-opiskelijalta tämä on jopa odotettavaa.

!!! warning "Jupyter Hub ja Colab"

    Jupyter Hub ei ymmärrä Marimo-notebookeja, joten ne täytyy ensin muuntaa Jupyter Notebook -muotoon. Sama pätee ainakin kirjoitushetkellä Colabiin. 
    
    Käännös onnistuu `marimo export ipynb notebook.py -o notebook.ipynb` -komennolla. Lue lisää Marimon [Coming from Jupyter](https://docs.marimo.io/guides/coming_from/jupyter/) -ohjeesta.

    On suositeltavaa kuitenkin opetella Marimo-työkalun käyttö.

