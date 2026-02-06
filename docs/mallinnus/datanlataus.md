---
priority: 410
---

# Datan lataus

## Datan lataus PyTorchissa

PyTorch tarjoaa `torch.utils.data`-moduulissa kaksi luokkaa, jotka helpottavat datan käsittelyä: `Dataset` ja `DataLoader`. Lisäksi löytyy esimerkiksi `TensorDataset`-luokka, josta voi olla apua harjoitellessa.

* **Dataset** on luokka, joka käärii sisäänsä datan ja labelit.
* **DataLoader** on luokka, joka käärii yllä olevan iteroitavaksi objektiksi.

### TensorDataset

Käytännössä voimme siis käyttää mitä tahansa tensoreita ja kääriä ne datasetiksi, näin:

```python
# 5 samples with 3 features each
X = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
    [1.5, 2.5, 3.5],
    [4.5, 5.5, 6.5]
])

# Binary labels
y = torch.tensor([0, 1, 1, 0, 1])

# Dataset
dataset = TensorDataset(X, y)
```

Nyt voimme kaivaa datasetistä yksittäisiä näytteitä:

```python
data, label = dataset[0]
# tensor([1.0, 2.0, 3.0])
# tensor(0)
```

### Tee-se-itse

Yksinkertaisimmillaan `Dataset`-luokan tarvitsee toteuttaa kaksi metodia: `__len__` ja `__getitem__`, joista tosin ensimmäinen on *optional*. Jos alaviivoilla ympäröidyt funktiot ovat sinulle vieraita, niin ne ovat Pythonin erikoismetodeja, jotka vastaavat, kun objektia kutsutaan `len(dataset)`-funktiolla tai objektia yritetään viipaloida (*engl. slice*) hakasulkeilla `dataset[idx]` tai `dataset[start:stop]`.

```python
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
```

Luonnollisesti voit tehdä datasetistä aivan mitä tahansa. Vain ohjelmointitaitosi ovat rajana. Voit toteuttaa esimerkiksi: 

* Etsi labelit indeksitiedostosta (`data/labels.txt`)
* Lue kuvat kovalevyltä (`data/train.parquet`, ...)
    * Tai kenties lataa S3:sta (jos `download=True`)
    * Tai varoita jos tuoreempi data on saatavilla verkosta
* Esikäsittele dataa lennossa (jos `transform`-parametri on määritelty)
* Palauta vain jokin tietty versio/subset datasta (esim. `train=True`)
* Parametri `obj.classes` sisältäen ihmisluettavat luokat
* Parametri `obj.class_to_idx` määrittelee, mikä luokka vastaa mitäkin indeksiä


### Jako koulutus- ja testidatasettiin

Yleinen käytäntö on jakaa data koulutus- ja testidatasettiin. Tämä onnistuu helposti `torch.utils.data.random_split`-funktiolla [^mlforcoders]:

```python
from torch.utils.data import random_split
 
ds = CustomDataset(...)

total_count = len(ds)
train_count = int(0.7 * total_count)
val_count = int(0.15 * total_count)
 
# Varmistetaan että kaikki näytteet tulevat käyttöön
# eli loput 15 %
test_count = total_count - train_count - val_count  
 
train_ds, val_ds, test_ds = 
     random_split(ds, [train_count, val_count, test_count])
```


### DataLoader

DataLoader on vain wrapper Datasetin ympärille, joten jatketaan yllä luodun `TensorDataset`-esimerkin parissa. Huomaa, että DataLoader ei toteuta `__getitem__`-metodia, vaan ainoastaan `__iter__`-metodin, joka palauttaa iteraattorin. Et voi siis hakea yksittäistä näytettä `mydataloader[0]`, vaan sinun tulee iteroida datan läpi esimerkiksi `for`-silmukassa tai `next()`-funktiolla.

```python
# Alustetaan aiemmin luodun datasetin pohjalta DataLoader
dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

# Loopataan ja tulostetaan datanäytteet
for data, label in dataloader:
    print(data.shape)
    print(label.shape)
    print()
```

```console title="Output"
torch.Size([2, 3])
torch.Size([2])

torch.Size([2, 3])
torch.Size([2])

torch.Size([1, 3])
torch.Size([1])
```

## Data ja PyTorch Vision

Myös TorchVision, joka on PyTorchin virallinen lisäkirjasto kuvankäsittelyyn, tarjoaa työkaluja datan lataukseen. Oletkin jo käyttänyt `torchvision.datasets.MNIST`-datasettiä aiemmissa harjoituksissa. Näiden lisäksi TorchVision tarjoaa pohjaluokkia (ks. [Base classes for custom datasets](https://docs.pytorch.org/vision/main/datasets.html#base-classes-for-custom-datasets)), jotka ovat: `DatasetFolder` ja `ImageFolder` ja `VisionDataset`. 

### VisionDataset

VisionDataset on Datasetin kaltainen pohjaluokka, joka tarjoaa lisätoiminnallisuutta kuvadatasetin käsittelyyn. Näistä näkyvimmät ovat:

* `root`: Datasetin juurihakemisto (käytetään tulostukseen)
* `transforms`: kutsuttava funktio, joka ottaa vastaan kuvat ja labelit ja palauttaa muokatun version niistä
* `transform`: kutsuttava funktio, joka ottaa vastaan vain kuvan ja palauttaa muokatun version siitä

Oletkin jo käyttänyt `transforms`-ominaisuutta aiemmin:

```python
# 110_first_model.py
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])
```

Itse luokan käyttö hoituu samalla tavalla kuin aiemmin esitellyn `CustomDataset`-esimerkin kanssa. Eli `class MyVisionDataset(VisionDataset):` ja toteutat `__len__` ja `__getitem__`-metodit.

### ImageFolder

ImageFolder on VisionDatasetin aliluokka, joka olettaa datan olevan järjestettynä hakemistoihin siten, että jokainen alihakemisto vastaa yhtä luokkaa ==aakkosellisessa järjestyksessä==. Tämä tekee kuvien lataamisesta helppoa, jos ne on järjestetty valmiiksi oikeisiin hakemistoihin. Esimerkiksi:

```
data/
    aasiankultakissa/
        001.png
        002.png
        ...
    aavikkoilves/
        001.png
        ...
    ...
    zorilla/
        123.png
        ...
```

Saisit nämä ladattua suoraan datasetiksi näin:

```python
from torchvision.datasets import ImageFolder

dataset = ImageFolder(root='data/')
```

Moroney vihjaa kirjassa AI and ML for Coders in PyTorch [^mlforcoders], että aakkosjärjestyksen tuomilta ongelmilta voi välttyä käyttämällä `target_transform`-parametria, jolla voi määritellä oman funktionsa labelien muokkaamiseen. Esimerkiksi:

```python
# Mikä tahansa luokitusjärjestys
custom_class_to_idx = {
    'aasiankultakissa': 42, 
    'aavikkoilves': 3, 
    ...,
    'zorilla': 1024
}

# Luodaan dataset, jonka target_transform on lambda-funktio
dataset = ImageFolder(
  root='data/',
  target_transform=
    lambda x: custom_class_to_idx[dataset.classes[x]]
)

# Ylikirjoitetaan class_to_idx
dataset.class_to_idx = custom_class_to_idx
print(dataset.class_to_idx)
```

## Muut keinot datan lataukseen

### Keras 3 ja PyTorch-backend

Kuten [PyTorch-luvussa](../tensorit/pytorch.md) mainittiin, Keras 3 tukee jälleen useita taustajärjestelmiä – mukaan lukien PyTorchia. Tämä tarkoittaa, että voit käyttää Kerasin tuttuja datasettejä ja datan käsittelytyökaluja, vaikka malli pyörisikin PyTorch-backendilla. Emme käytä tässä toteutuksessa Kerasia, mutta jätetään kuitenkin maininnan tasolle, että saatat törmätä tämän sortin snippetteihin:

```python
train_ds, val_ds = keras.utils.image_dataset_from_directory(
    "PetImages",
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
``` 

### Hugging Face Datasets

Tähän oletkin jo törmännyt `214_cyberhate.py`-tiedostossa. Hugging Facesta voi ladata Datasets-repoista dataa joko `huggingface-hub` tai `datasets`-kirjastolla. Hub-kirjasto on tarkoitettu myös muiden kuin datasettien noutamiseen: sillä voi etsiä Hugging Facesta esimerkiksi valmiiksi koulutettuja malleja.

Löydät näiden dokumentaation online: [Datasets](https://huggingface.co/docs/datasets/en/index), [huggingface_hub](https://github.com/huggingface/huggingface_hub).

#### huggingface-hub

Tämä on kyberviha-Notebookista tuttu:

```python
filepath = hf_hub_download(
    repo_id="sourander/yskills",
    repo_type="dataset",
    filename="ySKILLS_longitudinal_dataset_teacher_processed.csv",
)

df = pd.read_csv(filepath)
```

Ideaalitilanteessa CSV-tiedosto ladattaisiin jonkin oman PyTorch Datasets-toteutuksen avulla sisään. Tiedosto on kuitenkin niin pieni, että harjoituksessa se otettiin `train_test_split`:n jälkeen käyttöön `torch.FloatTensor(X_train).to(device)`-komennolla.

#### datasets

Yllä mainitulla työkalulla on näppärä ladata yksittäinen tiedosto kerrallaan. Jos Hugging Face -repo on oikein järjestetty, sieltä voi ladata kerralla train, test ja muutkin datasetit näin:

```python
from datasets import load_dataset
ds = load_dataset("username/repoid")
```

Esimerkiksi `sourander/yskills`-datasetti on järjestetty siten, että erillistä train/test/validation jakoa ei ole tehty tiedostotasolla vaan koko datasetti on samassa CSV-tiedostossa. Tällöin `load_dataset` lataa koko datasetin `train`-avaimeen. Jos `datasets`-kirjaston kanssa haluaisi kuitenkin edetä, voisi koodin näyttää tältä:

```python
from datasets import load_dataset

# CSV:n sisältö löytyy "train"-avaimesta.
# Tässä datasetissä ei ole test- tai validation-osioita.
ds = load_dataset("sourander/yskills").with_format("torch")

# Nyt on batch ja train
split_ds = ds["train"].train_test_split(test_size=0.3, seed=42)

# Siirretään ne eri loadereihin
trainloader = DataLoader(ds["train"], batch_size=32)
testloader = DataLoader(ds["test"], batch_size=32)

# Kaivetaan yksi batch ulos
for batch in trainloader:
    break
```

Tässä välissä on hyvä huomauttaa, että jatkossa `len(batch) != 32`, koska Hugging Face -Datasets palauttaa sanakirjan, jossa on avaimina sarakkeiden nimet. Eli siis `batch.keys()` palauttaa sarakenimet. Tämä vaatisi jotakuinkin seuraavanlaista käsittelyä koulutusloopissa:

```python
for batch in dataloader:
    # Erotellaan piirteet ja labelit ja stäkätään horisontaalisesti X:ksi
    feature_cols = [c for c in batch.keys() if c != "RISK101"]
    X = torch.stack([batch[col] for col in feature_cols], dim=1).float().to(device)
    y = batch["RISK101"].float().to(device)

    # Normaali forward pass
    outputs = model(X).squeeze(1)
```

Tämä on hieman hankalampi tapa edetä kuin suora CSV:n lukeminen Pandasilla, mutta on hyvä tunnistaa eri vaihtoehdot. Eri tutoriaaleissa ja eri kirjastojen kanssa pelatessa saatat törmätä yllättävään valikoimaan erilaisia tapoja datan lataukseen (ja muuhunkin).

## Tehtävät

!!! question "Tehtävä: MNIST MLP Revisited"

    Muistat varmaan aiemman `110_first_model.py`-notebookin, jossa ajoit ensimmäistä kertaa yksinkertaista eteenpäin kytkeytyvää neuroverkkoa MNIST-datasetillä. Palaa tähän harjoitukseen ja tee seuraavat parannukset:

    ```python
    import os
    NUM_CPU = os.cpu_count()

    # Voit kokeilla, hyötyykö GPU vaiko CPU enemmän:
    USE_GPU = True # or False

    # Lisää kumpaankin DataLoaderiin seuraavat keyword-argumentit:
    trainloader = DataLoader(..., persistent_workers=True, num_workers=NUM_CPU)
    testloader = DataLoader(..., persistent_workers=True, num_workers=NUM_CPU)
    ```

    Jos et ole kirjannut aiemmin koulutuksen kestoa ylös, tee se nyt ennen muutoksia. Tämän jälkeen tee yllä mainitut muutokset ja aja koulutus uudestaan. Kuinka paljon nopeutusta sait aikaan? Tutustu siihen, mitä `persistent_workers` ja `num_workers` tekevät ja miksi ne nopeuttavat datan latausta näinkin paljon.

!!! question "Tehtävä: CIFAR10 Datasetin plärääminen"

    Tutustu [CIFAR10](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html) -datasettiin. Datasetti koostuu 60 000 ==värikuvasta==. Kuvia on 6,000 kutakin luokkaa kohden eli luokkia on 10. Harjoittele seuraavia:

    * Lataa CIFAR10-datasetti PyTorchin `torchvision.datasets`-moduulista.
    * Tutki datan rakennetta (kuinka monta kuvaa, kuvan koko, värit, luokat).
    * Visualisoi yksi kuva
    * Visualisoi useampi kuva ruudukossa (grid). 
        * Bonus: Käyttäjä voi valita luokan.
    * Luo `DataLoader`, jolla voit iteroida datan läpi mini-batcheina.
        * Visualisoi yksi mini-batch ruudukossa.
  
    Voit käyttää apuna `410_cifar10.py`-notebookia tai kirjoittaa koodin alusta itse.


!!! question "Tehtävä: CIFAR10 Malli"

    Kouluta myös yksinkertainen eteenpäin kytkeytyvä (feedforward) neuroverkko CIFAR10-datasetillä. Tulemme myöhemmin vertaamaan tätä suoritusta konvoluutioverkkoon (CNN).

    Alta saat jo osviittaa, mihin lukemiin tulet pääsemään ==seuraavissa luvuissa==. Nyt voit olla tyytyväinen, jos pääset noin 50–60 % tarkkuuteen (accuracy). Mikä mahtaa olla baseline, jotta olet arpaa parempi?

    | Paper title                                                                | Error rate | Accuracy | Year |
    | -------------------------------------------------------------------------- | ---------- | -------- | ---- |
    | Convolutional Deep Belief Networks on CIFAR-10                             | 21.1 %     | 78.9%    | 2010 |
    | Maxout Networks                                                            | 9.38 %     | 90.62%   | 2013 |
    | Wide Residual Networks                                                     | 4.0 %      | 96.0%    | 2016 |
    | ...                                                                        | ...        | ...      | ...  |
    | An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale | 0.5 %      | 99.5%    | 2021 |

    Kannattaa tutkia myös, kauan mallin koulutus kestää. Data sekä malli ovat aiempaa suurempia. Opettajan valitsemalla arkkitehtuurilla saatiin 57 % testitarkkuus (joka ei juuri parantunut epookin nro 30 jälkeen). Tällä arkkitehtuurilla koulutus kesti, kun `persistent_workers=True` ja `num_workers=NUM_CPU` asetukset olivat paikoillaan:

    * CPU 100 epookkia: 10 min 5 s
    * GPU 100 epookkia: 8 min 20 s

    Taulukon lähde: [Wikipedia: CIFAR-10](https://en.wikipedia.org/wiki/CIFAR-10)

    Voit käyttää apuna `411_cifar10.py`-notebookia tai kirjoittaa koodin alusta itse.

!!! question "Tehtävä: ImageFolderin käyttö"

    Käytä ImageFolder-luokkaa ja `custom_class_to_idx`-sanakirjaa. Tähän on valmis pohja, jossa tarvii muokata vain yhtä solua. Katso `412_imagefolder.py`-notebook.

    Varmista, että ymmärrät, missä tiedostot ovat fyysisesti levyllä.

## Lähteet

[^mlforcoders]: Moroney, L. *AI and ML for Coders in PyTorch*. O'Reilly. 2025.
