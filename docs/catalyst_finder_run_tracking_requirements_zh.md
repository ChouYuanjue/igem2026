# Catalyst Finder 模型运行记录需求

## 一、目标

记录用户一次模型任务的完整过程，以便知道：

- 用户选择了哪个功能；
- 用户最终输入了什么；
- 用户测试的底物、产物、酶等对象是什么；
- 系统是否正确解析并找到对应反应；
- 模型使用了什么路线和版本；
- 模型返回了哪些候选酶；
- 用户对哪些候选感兴趣；
- 后续是否产生了湿实验结果。

本需求只记录和模型任务直接相关的信息，**不做全面的用户行为追踪**。

## 二、记录单位

需要建立三层 ID：

```text
session_id：一次用户访问会话
run_id：一次完整模型任务
step_id：该任务中的具体步骤
```

示例：

```text
session_id
└── run_id
    ├── step 1：点击快捷卡片
    ├── step 2：提交最终提示词
    ├── step 3：解析用户输入
    ├── step 4：检索反应
    ├── step 5：推荐候选酶
    └── step 6：用户对候选酶进行反馈
```

## 三、记录范围

### 1. 快捷卡片

用户点击快捷卡片时，记录：

```text
card_id
card_title
prompt_template
clicked_at
```

用途：确定用户选择的是哪个功能。

### 2. 最终提示词

用户点击“运行模型”时，记录：

```text
final_user_prompt
prompt_source
edited_after_card_click
submitted_at
```

必须同时保存：

- 快捷卡片原始提示词 `prompt_template`；
- 用户最后实际提交的完整提示词 `final_user_prompt`。

用户可能点击快捷卡片后继续修改内容，因此这两个字段不能合并。

### 3. 用户输入对象

从最终提示词中解析并保存：

```text
substrates      底物
products        产物
enzymes         用户指定的酶
organisms       生物种属
hosts           表达宿主
cofactors       辅因子
reaction_ids    反应 ID
protein_ids     蛋白 ID
other_entities  其他对象
```

每个对象最好同时保存：

```text
raw_text          用户原文
normalized_name   标准化名称
database_id       数据库 ID，如果有
resolution_status 匹配状态
```

示例：

```json
{
  "raw_text": "青蒿醛",
  "normalized_name": "artemisinic aldehyde",
  "database_id": null,
  "resolution_status": "matched"
}
```

### 4. 模型处理过程

记录模型任务中真正重要的处理步骤：

```text
intent_classification
entity_resolution
reaction_search
reaction_confirmation
candidate_ranking
route_design
pathway_analysis
```

每一步作为一次测试记录中的 `steps` 数组元素保存：

```text
step_id
step_type
status
input
output
error
started_at
finished_at
latency_ms
```

状态至少区分：

```text
success
no_match
dependency_unavailable
validation_failed
user_cancelled
model_failed
timeout
system_error
```

“没有找到反应”和“系统报错”必须分开记录。

### 5. 模型推荐结果

保存模型返回的完整候选结果，而不是只保存 Top 1：

```text
candidate_id
candidate_rank
candidate_score
score_components
known_or_novel
evidence
warnings
```

同时保存结果当时的版本信息：

```text
app_version
model_version
registry_version
route_id
data_version
```

这样以后才能复现和解释当时的推荐结果。

## 四、候选酶快速反馈

候选酶列表中可以提供：

```text
有意思
没兴趣
不确定
```

用户点击后直接提交，不要求填写文字。

记录：

```text
run_id
candidate_id
candidate_rank
candidate_score
signal
submitted_at
```

这个反馈表示用户的初步兴趣，不能解释为实验成功。

## 五、湿实验结果

湿实验反馈单独保存，不与快速兴趣反馈混合。

用户点击“记录实验结果”后，打开独立窗口，填写：

```text
experiment_id
run_id
candidate_id
tested
expression_status
activity_status
product_detected
conversion_rate
yield
replicate_count
conditions
notes
submitted_at
```

建议的结果状态：

```text
not_tested
planned
expression_failed
assay_failed
no_target_product
weak_activity
active
high_activity
inconclusive
```

需要明确：

```text
用户点击“有意思”不等于实验有效；
模型评分高不等于酶一定可用。
```

## 六、统一关联关系

每次模型任务生成一个 `run_id`：

```text
run_id
├── 快捷卡片记录
├── 最终提示词
├── 用户实体解析
├── 反应检索结果
├── 模型推荐结果
├── 候选兴趣反馈
└── 湿实验结果
```

候选酶使用 `candidate_id` 关联，实验使用 `experiment_id` 关联。

这样可以追踪：

> 某个酶是哪个用户任务中被推荐的，模型当时给了第几名，用户是否感兴趣，后来有没有做实验，实验结果是什么。

## 七、明确不记录的内容

当前不需要记录：

```text
鼠标移动
每次键盘输入
用户输入过程中的每个字符
打开候选详情
展开证据
打开流程图
普通页面点击
```

也不需要保存用户个人信息：

```text
姓名
邮箱
手机号
密码
API Key
完整 IP
Cookie
浏览器指纹
```

只记录与模型任务、候选判断和实验结果有关的数据。

## 八、存储方案

第一版可以使用两个 JSONL 文件：

```text
results/catalyst_finder_runtime/run_events.jsonl
results/catalyst_finder_runtime/wet_lab_results.jsonl
```

其中：

- `run_events.jsonl`：一条 JSONL 记录对应一次完整模型测试；快捷卡片、最终提示词、模型结果和各处理步骤统一保存在同一条记录中，步骤放在 `steps` 数组内；
- `wet_lab_results.jsonl`：湿实验结果。

`run_events.jsonl` 不是“一步一行”。一次完整测试通常只有一个 `event_id`，通过其中的 `steps` 数组保留解析、检索和排序过程。

文件只允许服务账号读取，不提供公网查询接口。

如果后续需要按会话、反应类型或候选酶进行复杂查询，再迁移到 SQLite 或 PostgreSQL。

## 九、开发顺序

```text
第一步：增加 run_id、step_id 和事件记录结构
第二步：记录快捷卡片和最终完整提示词
第三步：记录底物、产物、酶等实体解析结果
第四步：记录反应检索和模型推荐结果
第五步：增加候选酶“有意思/没兴趣/不确定”按钮
第六步：增加独立的湿实验结果窗口
第七步：增加管理员查看和统计命令
```

## 十、验收标准

使用一次完整测试任务后，系统应该能够回答：

1. 用户点击了哪个快捷卡片？
2. 快捷卡片原始提示词是什么？
3. 用户最后提交的完整提示词是什么？
4. 用户输入了哪些底物、产物和酶？
5. 系统是否正确识别这些对象？
6. 是否找到对应反应？
7. 模型使用了哪条路线和哪个版本？
8. 返回了哪些候选酶及其排名？
9. 每一步是成功、无匹配还是报错？
10. 用户对哪些候选点击了“有意思”？
11. 哪些候选后来产生了湿实验结果？

## 十一、总结

我们只记录与模型任务直接相关的关键数据：快捷卡片、最终提示词、实体解析、模型处理结果、候选兴趣反馈和湿实验结果，不记录无关的页面点击。

目标是以 `run_id` 为主线，完整记录用户从选择功能、提交提示词、解析底物和产物、获得候选酶，到兴趣判断和湿实验验证的过程。
