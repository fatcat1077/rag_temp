# Trust RAG 系統文件大綱

## 1. 專案簡介
- 這套系統提供 RAG（檢索增強生成）問答 API
- 模式：`articles`（文章） / `products`（產品）
- 所有回答都會同時檢索 `laws`（法規）作為依據

## 2. 下載專案（GitHub）
- clone 專案
- 進入專案資料夾

## 3. 環境安裝與設定
- 建立/啟用虛擬環境（.venv）
- 安裝 requirements
- 建立 `.env` 並設定 `OPENAI_API_KEY`

## 4. 資料夾結構
- `data/laws/`：法規文本
- `data/articles/`：文章文本
- `data/products/`：產品文本
- `db/`：向量庫輸出（由 ingest 產生，不上傳 GitHub）

## 5. 建庫流程（ingest）
- 執行 `ingest_all.py`
- 產生三個 collection：`trust_laws` / `trust_articles` / `trust_products`

## 6. 啟動服務（app）
- 執行 `app.py`
- 提供 `POST /ask` API

## 7. API 使用方式
- Request：`mode`、`question`（可選 `user_profile`、`main_top_k`、`laws_top_k`）
- Response：`answer`、`sources`、`meta`

## 8. 新增/更新資料與重建向量庫
- 將新文字放入對應資料夾（laws/articles/products）
- 修改資料後需重新執行 ingest（必要時先刪除 `db/`）

## 9. 常見問題（可選）
- `.env` 沒讀到 / 金鑰錯誤
- `chunks=0`（文本為空或編碼問題）
- 忘記先 ingest 導致查不到資料