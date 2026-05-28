CREATE DATABASE IF NOT EXISTS `packing_demo` CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;
USE `packing_demo`;

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;
DROP TABLE IF EXISTS `anomaly_log`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `anomaly_log` (
  `id` int NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `created_at` datetime DEFAULT NULL COMMENT '记录创建时间（北京时间）',
  `anomaly_type` varchar(50) DEFAULT NULL COMMENT '异常类型分类（如 api_error / timeout / overload）',
  `severity` varchar(20) DEFAULT NULL COMMENT '严重级别（warning / error / critical）',
  `session_id` varchar(100) DEFAULT NULL COMMENT '关联的计算会话ID',
  `calc_no` varchar(50) DEFAULT NULL COMMENT '关联的计算编号（格式 YYYYMMDD-NNN）',
  `error_code` varchar(50) DEFAULT NULL COMMENT '错误码',
  `error_msg` text COMMENT '错误详细信息',
  `duration_ms` int DEFAULT NULL COMMENT '操作耗时（毫秒）',
  `extra` json DEFAULT NULL COMMENT '附加扩展信息（JSON格式）',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;
DROP TABLE IF EXISTS `bins_catalog`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `bins_catalog` (
  `id` int NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `sku` varchar(50) NOT NULL COMMENT '包材SKU编号',
  `name` varchar(1000) NOT NULL COMMENT '包材完整名称及规格描述',
  `price` decimal(10,2) DEFAULT '0.00' COMMENT '包材单价（元）',
  `length` decimal(10,1) NOT NULL COMMENT '包材内径长度（cm）',
  `width` decimal(10,1) NOT NULL COMMENT '包材内径宽度（cm）',
  `height` decimal(10,1) NOT NULL COMMENT '包材内径高度（cm），袋子类平铺时为0',
  `mat_weight` decimal(10,3) DEFAULT '0.000' COMMENT '包材自重（kg）',
  `bin_type` varchar(20) NOT NULL COMMENT '包材大类：硬包材 / 软包材 / 辅材',
  `mat_type` varchar(50) NOT NULL COMMENT '包材小类：三层纸箱 / 纸箱 / 单层纸箱 / 袋子 / 辅材',
  `is_custom` char(1) NOT NULL DEFAULT '否' COMMENT '是否定制包材：是 / 否',
  `protection_level` varchar(10) NOT NULL COMMENT '防护等级文字：一级 / 二级 / 三级 / 四级',
  `protection_rank` tinyint NOT NULL COMMENT '防护等级数值（1-4，值越大保护越好）',
  `max_weight` decimal(10,3) DEFAULT '22.000' COMMENT '最大承重（kg），默认22',
  PRIMARY KEY (`id`),
  KEY `idx_bin_type` (`bin_type`),
  KEY `idx_mat_type` (`mat_type`),
  KEY `idx_protection_rank` (`protection_rank`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;
DROP TABLE IF EXISTS `feedback`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `feedback` (
  `result_id` varchar(64) NOT NULL COMMENT '关联的装箱结果ID（对应pack_results.result_id）',
  `session_id` varchar(36) DEFAULT NULL COMMENT '关联的计算会话ID（同一次计算多个方案共享）',
  `calc_no` varchar(20) DEFAULT NULL COMMENT '计算编号（格式 YYYYMMDD-NNN）',
  `plan_no` varchar(22) DEFAULT NULL COMMENT '方案编号（格式 calc_no-P1/P2/P3）',
  `plan_type` varchar(50) DEFAULT NULL COMMENT '包材类型标签（纸箱 / 袋子等，对应前端方案分类）',
  `recommended_bin` varchar(100) DEFAULT NULL COMMENT '系统推荐的箱型名称',
  `recommended_sku` varchar(100) DEFAULT NULL COMMENT '系统推荐的包材SKU编号',
  `selected_plan` varchar(20) DEFAULT NULL COMMENT '操作员最终选择的方案：rec=推荐新包材 / soft=软包材 / best=包材库最优',
  `selection_method` varchar(20) DEFAULT NULL COMMENT '选择方式：default=系统默认 / auto=静默自动 / manual=手动选择',
  `adopted` tinyint(1) DEFAULT NULL COMMENT '是否采纳推荐：1=已采纳 0=未采纳 NULL=未反馈',
  `actual_used_bin` varchar(100) DEFAULT NULL COMMENT '操作员实际使用的箱型名称（未采纳时填写）',
  `actual_used_sku` varchar(100) DEFAULT NULL COMMENT '操作员实际使用的包材SKU（未采纳时填写）',
  `reason_changed` varchar(200) DEFAULT NULL COMMENT '未采纳原因分类，如：尺寸不合适 / 库存不足 / 其他',
  `reason_detail` text COMMENT '未采纳原因详细说明（自由文本）',
  `items_summary` json DEFAULT NULL COMMENT '装箱货品摘要（JSON数组，存储反馈时的货品快照）',
  `operator_id` varchar(64) DEFAULT NULL COMMENT '操作员ID或工号',
  `created_at` varchar(20) DEFAULT NULL COMMENT '反馈创建时间（北京时间，格式YYYY-MM-DD HH:MM:SS）',
  `updated_at` varchar(20) DEFAULT NULL COMMENT '反馈最后更新时间（北京时间，格式YYYY-MM-DD HH:MM:SS）',
  `selected_rank` tinyint DEFAULT NULL COMMENT '选择的现有包材排名1/2/3',
  PRIMARY KEY (`result_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;
DROP TABLE IF EXISTS `optimization_feedback`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `optimization_feedback` (
  `id` int unsigned NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `result_id` varchar(64) DEFAULT NULL COMMENT '关联的装箱结果ID（对应 pack_results.result_id）',
  `category` varchar(50) DEFAULT NULL COMMENT '反馈分类（如：计算结果 / 包材推荐 / 其他）',
  `content` text COMMENT '反馈内容详情',
  `operator_id` varchar(50) DEFAULT NULL COMMENT '操作员ID或工号',
  `created_at` varchar(20) DEFAULT NULL COMMENT '反馈创建时间（北京时间，格式 YYYY-MM-DD HH:MM:SS）',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;
DROP TABLE IF EXISTS `pack_result_input_bins`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `pack_result_input_bins` (
  `id` int unsigned NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `result_id` varchar(64) NOT NULL COMMENT '关联的装箱结果ID（对应pack_results.result_id）',
  `type` varchar(100) NOT NULL DEFAULT '' COMMENT '用户手填的包材箱型名称或编号',
  `length` decimal(10,2) NOT NULL COMMENT '包材内径长度（cm）',
  `width` decimal(10,2) NOT NULL COMMENT '包材内径宽度（cm）',
  `height` decimal(10,2) NOT NULL COMMENT '包材内径高度（cm）',
  `max_weight` decimal(10,3) NOT NULL COMMENT '包材最大承重（kg）',
  PRIMARY KEY (`id`),
  KEY `idx_result_id` (`result_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;
DROP TABLE IF EXISTS `pack_result_items`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `pack_result_items` (
  `id` int unsigned NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `result_id` varchar(64) NOT NULL COMMENT '关联的装箱结果ID（对应pack_results.result_id）',
  `item_id` varchar(64) NOT NULL DEFAULT '' COMMENT '前端传入的商品ID',
  `length` decimal(10,2) NOT NULL COMMENT '商品长度（cm）',
  `width` decimal(10,2) NOT NULL COMMENT '商品宽度（cm）',
  `height` decimal(10,2) NOT NULL COMMENT '商品高度（cm）',
  `weight` decimal(10,3) NOT NULL COMMENT '商品重量（kg）',
  `product_title` varchar(255) NOT NULL DEFAULT '' COMMENT '商品品名或标题，供AI判断防护级别',
  `sale_price` decimal(10,2) NOT NULL DEFAULT '0.00' COMMENT '商品售价（USD），用于运费档级计算',
  `product_category` varchar(50) NOT NULL DEFAULT '' COMMENT '产品分类：常规类产品 / 服装产品 / 危险品',
  `soft_packaging_ok` tinyint(1) NOT NULL DEFAULT '0' COMMENT '是否允许使用软包材：1=允许 0=不允许',
  PRIMARY KEY (`id`),
  KEY `idx_result_id` (`result_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;
DROP TABLE IF EXISTS `pack_results`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `pack_results` (
  `result_id` varchar(64) NOT NULL COMMENT '装箱计算结果唯一ID（UUID）',
  `session_id` varchar(36) DEFAULT NULL COMMENT '计算会话ID（同一次计算多个方案共享）',
  `calc_no` varchar(20) DEFAULT NULL COMMENT '计算编号（格式 YYYYMMDD-NNN，同会话内唯一）',
  `plan_no` varchar(22) DEFAULT NULL COMMENT '方案编号（格式 calc_no-P1/P2/P3）',
  `plan_type` varchar(50) DEFAULT NULL COMMENT '包材类型标签（纸箱 / 袋子等，对应前端方案分类）',
  `winner` varchar(200) DEFAULT NULL COMMENT '推荐结果标签（推荐新包材 / 包材库最优 / 软包材等）',
  `winner_bin` varchar(100) DEFAULT NULL COMMENT '推荐方案对应的箱型名称（软包材时为尺寸描述）',
  `winner_sku` varchar(100) DEFAULT NULL COMMENT '推荐方案对应的包材SKU编号',
  `winner_tier` varchar(30) DEFAULT NULL COMMENT '推荐方案对应的FBA运费档级',
  `winner_total_fee` decimal(10,2) DEFAULT NULL COMMENT '推荐方案的FBA总费用（USD）',
  `existing_bin` varchar(200) DEFAULT NULL COMMENT '包材库当前最优方案的箱型名称',
  `existing_tier` varchar(30) DEFAULT NULL COMMENT '原包材库最优方案的FBA运费档级',
  `existing_total_fee` decimal(10,2) DEFAULT NULL COMMENT '原包材库最优方案的FBA总费用（USD）',
  `tier_upgraded` tinyint(1) DEFAULT NULL COMMENT '是否成功降档：1=是 0=否',
  `fee_saved` decimal(10,2) DEFAULT NULL COMMENT '相比原方案节省的FBA费用（USD），原费减推荐费',
  `utilization` decimal(5,4) DEFAULT NULL COMMENT '推荐箱型的空间利用率（0~1小数）',
  `item_count` int DEFAULT NULL COMMENT '本次装箱的商品总件数',
  `total_weight` decimal(10,3) DEFAULT NULL COMMENT '所有商品的总重量（kg）',
  `product_category` varchar(50) DEFAULT NULL COMMENT '产品分类：常规类产品 / 服装产品 / 危险品',
  `ai_used` tinyint(1) DEFAULT NULL COMMENT '是否使用AI决策：1=Claude AI 0=本地算法',
  `created_at` varchar(20) DEFAULT NULL COMMENT '记录创建时间（北京时间，格式YYYY-MM-DD HH:MM:SS）',
  `ai_model` varchar(60) DEFAULT NULL COMMENT 'AI模型名称',
  `ai_provider` varchar(50) DEFAULT NULL COMMENT 'AI厂商（anthropic / deepseek）',
  `ai_error` varchar(500) DEFAULT NULL COMMENT 'AI调用失败时的错误信息，正常时为空',
  `ai_input_tokens` int DEFAULT NULL COMMENT 'AI输入token数',
  `ai_output_tokens` int DEFAULT NULL COMMENT 'AI输出token数',
  `top3_existing_json` text COMMENT '包材库前3优方案JSON',
  `classify_input_tokens` int DEFAULT NULL COMMENT 'classify_for_packaging AI输入token',
  `classify_output_tokens` int DEFAULT NULL COMMENT 'classify_for_packaging AI输出token',
  `classify_source` varchar(10) DEFAULT NULL COMMENT 'ai=AI调用 keyword=本地规则',
  `classify_model` varchar(60) DEFAULT NULL COMMENT '防护分析使用的AI模型',
  `classify_provider` varchar(30) DEFAULT NULL COMMENT '防护分析AI厂商',
  PRIMARY KEY (`result_id`),
  KEY `idx_session_id` (`session_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;
DROP TABLE IF EXISTS `pack_scheme_detail`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `pack_scheme_detail` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT COMMENT '自增主键',
  `calc_no` varchar(32) DEFAULT NULL COMMENT '计算编号（格式 YYYYMMDD-NNN）',
  `plan_no` varchar(34) DEFAULT NULL COMMENT '方案编号（格式 calc_no-P1/P2/P3）',
  `session_id` varchar(64) DEFAULT NULL COMMENT '关联的计算会话ID',
  `plan_type` varchar(50) DEFAULT NULL COMMENT '包材类型标签（纸箱 / 袋子等）',
  `classify_result` json DEFAULT NULL COMMENT '产品防护分类结果（JSON：protection_level / suitable_types / reason）',
  `agent_summary` text COMMENT 'AI Agent生成的方案文字摘要',
  `final_result` longtext COMMENT '最终装箱结果详情（JSON：packed_bins / summary等）',
  `compare_result` longtext COMMENT '包材对比结果详情（JSON：推荐方案与现有方案对比数据）',
  `created_at` datetime DEFAULT NULL COMMENT '记录创建时间（北京时间）',
  `duration_ms` int DEFAULT NULL COMMENT '计算总耗时（毫秒）',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_calc_plan` (`calc_no`,`plan_no`),
  KEY `idx_calc_no` (`calc_no`),
  KEY `idx_plan_no` (`plan_no`),
  KEY `idx_session` (`session_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

