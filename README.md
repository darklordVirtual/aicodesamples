# AI og Integrasjoner - Kodeeksempler

Komplett kodebase for boken **"AI og Integrasjoner: Fra grunnleggende til avansert"** av Stian Skogbrott.

## 📚 Om Prosjektet

Dette repositoryet inneholder alle kodeeksemplene fra boken, organisert i en modulær og testbar struktur som følger beste praksis for produksjonsklar Python-kode.

## 🏗️ Prosjektstruktur

```
aicodesamples/
├── src/
│   ├── fundamentals/          # Del 1: Grunnleggende (Kap 1-3)
│   │   ├── ai_basics.py       # AI-klienter og grunnleggende bruk
│   │   ├── prompt_engineering.py
│   │   └── embeddings.py      # Embeddings og semantisk søk
│   │
│   ├── vector_db/             # Del 2: Vektordatabaser (Kap 4-5)
│   │   ├── chromadb_basics.py
│   │   ├── advanced_chromadb.py
│   │   └── chunking.py
│   │
│   ├── mcp/                   # Del 3: MCP (Kap 6-7)
│   │   ├── simple_server.py
│   │   ├── tripletex_server.py
│   │   └── tripletex_client.py
│   │
│   ├── integrations/          # Del 4: Avanserte integrasjoner (Kap 8-10)
│   │   ├── rag_system.py
│   │   ├── agents.py
│   │   └── production.py      # Feilhåndtering, retry, rate limiting
│   │
│   ├── optimization/          # Del 5: Optimalisering (Kap 11-12)
│   │   ├── cost_optimization.py
│   │   └── testing.py
│   │
│   ├── case_studies/          # Del 6: Case Studies (Kap 13-16)
│   │   ├── invoice_processing.py
│   │   ├── customer_support.py
│   │   ├── multimodal.py
│   │   └── ethics.py
│   │
│   └── utils/                 # Felles verktøy
│       ├── config.py
│       ├── logging_config.py
│       └── security.py
│
├── tests/                     # Tester
│   ├── test_fundamentals/
│   ├── test_vector_db/
│   ├── test_mcp/
│   ├── test_integrations/
│   └── test_case_studies/
│
├── examples/                  # Praktiske eksempler
│   ├── 01_basic_ai_query.py
│   ├── 02_chromadb_demo.py
│   ├── 03_rag_demo.py
│   └── 04_agent_demo.py
│
├── data/                      # Eksempeldata
├── docs/                      # Dokumentasjon
├── .env.example
├── requirements.txt
├── pytest.ini
└── README.md
```

## 🚀 Kom i Gang

### 1. Klon Repositoryet

```bash
git clone https://github.com/luftfiber/ai-integrasjoner-norsk.git
cd ai-integrasjoner-norsk
```

### 2. Opprett Virtuelt Miljø

```bash
python -m venv venv
source venv/bin/activate  # På Windows: venv\Scripts\activate
```

### 3. Installer Avhengigheter

```bash
pip install -r requirements.txt
```

### 4. Konfigurer Environment

```bash
cp .env.example .env
# Rediger .env og legg til dine API-nøkler
```

### 5. Verifiser Installasjonen

```bash
pytest tests/ -v
```

## 📖 Brukseksempler

### Grunnleggende AI-bruk

```python
from src.fundamentals.ai_basics import AIClient

client = AIClient()
response = client.query("Forklar hva en vektordatabase er")
print(response)
```

### Embeddings og Semantisk Søk

```python
from src.fundamentals.embeddings import EmbeddingService

service = EmbeddingService()
embedding = service.get_embedding("Fiberinstallasjon problemer")
similar = service.find_similar("Nettverksproblemer", n_results=5)
```

### ChromaDB

```python
from src.vector_db.chromadb_basics import KnowledgeBase

kb = KnowledgeBase()
kb.add_documents([
    "SLA garanterer 99.9% oppetid",
    "Installasjon tar 2-3 uker"
])
results = kb.search("Hva er oppetidsgarantien?")
```

### RAG System

```python
from src.integrations.rag_system import RAGSystem

rag = RAGSystem()
answer = rag.query("Hvordan installerer man fiber?")
print(answer['answer'])
print(answer['sources'])
```

### Agent System

```python
from src.integrations.agents import SimpleAgent

agent = SimpleAgent(tools={
    "get_customer": get_customer_func,
    "send_email": send_email_func
})

result = agent.run("Send påminnelse til kunde #123")
```

## 🧪 Kjøre Tester

```bash
# Alle tester
pytest

# Med coverage
pytest --cov=src --cov-report=html

# Spesifikk test-fil
pytest tests/test_fundamentals/test_ai_basics.py -v

# Kjør kun raske tester (ekskluderer integrasjonstester)
pytest -m "not integration"
```

## 📝 Kapittel-referanse

| Kapittel | Modul | Beskrivelse |
|----------|-------|-------------|
| 1-3 | `src/fundamentals/` | AI-grunnlag, prompt engineering, embeddings |
| 4-5 | `src/vector_db/` | ChromaDB og vektordatabaser |
| 6-7 | `src/mcp/` | Model Context Protocol og Tripletex |
| 8-10 | `src/integrations/` | RAG, agenter, produksjonssikring |
| 11-12 | `src/optimization/` | Kostnader og testing |
| 13-16 | `src/case_studies/` | Fakturabehandling, kundesupport, etikk |

## 🔒 Sikkerhet

- Alle API-nøkler skal lagres i `.env` (aldri commit denne!)
- Input-validering er implementert i `src/utils/security.py`
- Prompt injection-sjekker er aktivert by default
- GDPR-compliance følger `src/case_studies/ethics.py`

## 🎯 MCP-servere

### Kjøre Simple Customer Server

```bash
python src/mcp/simple_server.py
```

### Kjøre Tripletex MCP Server

```bash
python src/mcp/tripletex_server.py
```

Konfigurer i Claude Desktop (`~/.config/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "customer-db": {
      "command": "python3",
      "args": ["/full/path/to/src/mcp/simple_server.py"]
    },
    "tripletex": {
      "command": "python3",
      "args": ["/full/path/to/src/mcp/tripletex_server.py"]
    }
  }
}
```

## 📊 Performance

Systemet er optimalisert for:
- ⚡ Rate limiting (konfigurerbart)
- 💾 Intelligent caching
- 🔄 Automatisk retry med exponential backoff
- 📈 Batch-processing for store datamengder

## 🤝 Bidra

Bidrag er velkomne! Se [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 Lisens

MIT License - se [LICENSE](LICENSE)

## 📧 Kontakt

- **Forfatter**: Stian Skogbrott
- **Firma**: Luftfiber AS
- **GitHub**: [https://github.com/darklordVirtual/aicodesamples/](https://github.com/darklordVirtual/aicodesamples/)

## 🙏 Anerkjennelser

Takk til:
- Anthropic for Claude API
- OpenAI for embeddings API
- ChromaDB-teamet
- Tripletex for API-dokumentasjon

---

⭐ Hvis du finner dette nyttig, gi gjerne en star på GitHub!
