# Syväoppiminen I

!!! warning

    Tämä sivusto on kokonaisuudessaan työn alla. Valmistuu keväällä 2026.

## Kurssin kuvaus

Kurssinmateriaalin kirjoittanut opettaja ei ole matematiikan opettaja, mutta kurssilla esiintyy silti pieni määrä matematiikkaa. Matematiikka on selitetty ns. data engineerin näkökulmasta. Kurssilla pysytään intuition tasolla. Mitä matematiikkaa sitten AI:n kontekstissa tarvitaan? Yleisesti ottaen differentiaali- ja integraalilaskentaa, lineaarialgebraa, todennäköisyyslaskentaa ja tilastotiedettä. Joissakin AI:n osa-alueissa voi tulla tarvetta myös diskreetin matematiikan, optimoinnin perusteisiin, peliteorian tai matriisilaskennan perusteisiin. Tämän kurssin matematiikka on kuitenkin hyvin kevyttä. On kuitenkin suositeltavaa heijastella tämän kurssin sisältöjä opetussuunnitelman mukaisiin matematiikan opintoihin. Tutki, kuinka matematiikan harjoituksissa esiintyneet käsitteet näkyvät tässä kurssissa.

## Tehtävät

Kurssi sisältää tehtäviä, jotka on tarkoitettu tehtäviksi kurssin aikana. Tehtävät löytyvät kunkin osion lopusta. Lisäksi **kaikki** tehtävät ovat koostettuna [Tehtäväkooste](exercises.md)-sivulle. Tehtävät palautetaan [Oppimispäiväkirja 101](https://sourander.github.io/oat/) -ohjeistuksen mukaisesti eli Gitlab Pages:ssa hostattuna staattisena sivustona.

Useat tehtävät viittaavat [Marimo]-työkalulla tehtyihin notebookeihin. Kyseessä on Jupyter Notebook -työkalun seuraaja. Löydät notebookit [gh:sourander/syvaopiminen/notebooks](https://github.com/sourander/syvaoppiminen/tree/main/notebooks) -polusta *tämän kurssin* repositoriosta. Notebookien käyttö neuvotaan kurssivideoilla ja tukisessioissa.

## Koodin ajaminen

Kurssilla ei pelkästään katsella koodia vaan sitä ajetaan myös. Voit ajaa koodit 

* Jupyter Hubissa
* Google Colabiss
* Coder-ympäristössä
* Paikallisesti `uv`:lla

Vaihtoehtoja esitellään kurssivideoilla ja tukisessioissa. Mikäli sinulla on oma näytönohjain, on äärimmäisen suositeltavaa opetella ajamaan koodia paikallisesti kyseistä näytönohjainta hyödyntäen. AI-opiskelijalta tämä on jopa odotettavaa.

!!! warning "Jupyter Hub ja Colab"

    Jupyter Hub ei ymmärrä Marimo-notebookeja, joten ne täytyy ensin muuntaa Jupyter Notebook -muotoon. Sama pätee ainakin kirjoitushetkellä Colabiin. 
    
    Käännös onnistuu `marimo export ipynb notebook.py -o notebook.ipynb` -komennolla. Lue lisää Marimon [Coming from Jupyter](https://docs.marimo.io/guides/coming_from/jupyter/) -ohjeesta.

    On suositeltavaa kuitenkin opetella Marimo-työkalun käyttö.