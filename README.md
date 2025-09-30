# TopSun Dashboard Operacional

Dashboard integrado para análise de custos, tarefas e mapa de reparos/instalações.

## Funcionalidades Implementadas

### ✅ Funcionalidades Existentes (Mantidas)
- Análise de custos com upload de planilhas
- Filtros avançados por instalador, usuário, cliente, etc.
- Gráficos interativos de margem de lucro e lucratividade
- Tabela detalhada de ordens de serviço
- Mapa de reparos com geocodificação
- Busca de proximidade

### 🆕 Novas Funcionalidades Adicionadas
- **Upload de planilha de instalações** na aba do mapa
- **Extração automática** das colunas B (Projeto), D (Cliente), E (Cidade), F (Instalador), I (Data de previsão)
- **Filtro de tipo** no mapa: Reparos / Instalações / Todos
- **Ícones diferenciados**:
  - 🔧 Ferramenta para reparos
  - ⚡ Raio para instalações
- **Cores por instalador** mantidas para ambos os tipos
- **Integração completa** no dashboard e listas
- **Busca de proximidade** incluindo instalações

## Estrutura do Projeto

```
topsun-dashboard/
├── app.py                 # Backend Flask
├── templates/
│   └── index.html        # Frontend HTML/CSS/JS
├── static/
│   └── logo.png          # Logo da empresa
└── README.md             # Este arquivo
```

## Como Usar

### 1. Instalar Dependências
```bash
pip3 install flask flask-cors pandas openpyxl geopy
```

### 2. Executar o Servidor
```bash
python3 app.py
```

### 3. Acessar o Dashboard
Abra o navegador em: http://localhost:5000

### 4. Upload de Instalações
1. Vá para a aba "Mapa de Reparos"
2. Use o novo campo "Carregar Instalações em Aberto"
3. Faça upload da planilha Excel com as colunas:
   - B: Projeto
   - D: Cliente  
   - E: Cidade
   - F: Instalador
   - I: Data de previsão da instalação

### 5. Filtros e Visualização
- Use o filtro "Tipo" para ver apenas reparos, instalações ou ambos
- Os ícones no mapa diferenciam visualmente os tipos
- A tabela mostra badges coloridos para cada tipo
- A busca de proximidade inclui ambos os tipos

## Tecnologias Utilizadas

- **Backend**: Flask, Pandas, GeoPy
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Mapas**: Leaflet.js
- **Tabelas**: DataTables
- **Gráficos**: Chart.js

## Arquivos de Exemplo

Incluído: `ProjetosPendentesdeConclusãoFotosdaInstalação(24092025).xlsx`
