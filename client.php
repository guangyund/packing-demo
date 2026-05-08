<?php
/**
 * 三维装箱服务 PHP 客户端
 * 在 WMS 发货单生成节点调用装箱计算服务
 */

class BinPackingClient
{
    private string $serviceUrl;
    private int $timeout;

    public function __construct(string $serviceUrl = 'http://localhost:8000', int $timeout = 30)
    {
        $this->serviceUrl = rtrim($serviceUrl, '/');
        $this->timeout    = $timeout;
    }

    /**
     * 直接装箱计算（需指定箱型）
     *
     * @param array $items 货物列表
     * @param array $bins  箱型列表
     * @return array
     */
    public function pack(array $items, array $bins): array
    {
        return $this->post('/api/pack', [
            'items' => $items,
            'bins'  => $bins,
        ]);
    }

    /**
     * AI Agent 自动装箱（自动选箱型）
     *
     * @param array $items 货物列表
     * @return array
     */
    public function agentPack(array $items): array
    {
        return $this->post('/api/agent-pack', ['items' => $items]);
    }

    private function post(string $path, array $data): array
    {
        $ch = curl_init($this->serviceUrl . $path);
        curl_setopt_array($ch, [
            CURLOPT_POST           => true,
            CURLOPT_POSTFIELDS     => json_encode($data),
            CURLOPT_HTTPHEADER     => ['Content-Type: application/json'],
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_TIMEOUT        => $this->timeout,
        ]);

        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $error    = curl_error($ch);
        curl_close($ch);

        if ($error) {
            return ['success' => false, 'error' => "请求失败: {$error}"];
        }

        if ($httpCode !== 200) {
            return ['success' => false, 'error' => "服务返回错误: HTTP {$httpCode}"];
        }

        return json_decode($response, true) ?? ['success' => false, 'error' => '响应解析失败'];
    }
}


// ── 使用示例 ──────────────────────────────────────────────────────────────────

$client = new BinPackingClient('http://localhost:8000');

// 模拟从 WMS 数据库查出的出库货物
$items = [
    ['id' => 'SKU001-键盘',   'length' => 45, 'width' => 15, 'height' =>  5, 'weight' => 0.8],
    ['id' => 'SKU002-鼠标',   'length' => 12, 'width' =>  8, 'height' =>  4, 'weight' => 0.2],
    ['id' => 'SKU003-显示器', 'length' => 55, 'width' => 35, 'height' => 15, 'weight' => 4.5],
    ['id' => 'SKU004-耳机',   'length' => 20, 'width' => 18, 'height' => 10, 'weight' => 0.3],
    ['id' => 'SKU005-充电器', 'length' => 10, 'width' =>  6, 'height' =>  4, 'weight' => 0.2],
];

echo "=== 方式一：直接指定箱型 ===\n";
$bins = [
    ['type' => '中号箱', 'length' => 60, 'width' => 50, 'height' => 50, 'max_weight' => 30],
];
$result = $client->pack($items, $bins);
printResult($result);


echo "\n=== 方式二：AI Agent 自动选箱 ===\n";
$agentResult = $client->agentPack($items);

if ($agentResult['success'] ?? false) {
    echo "Agent 分析：\n" . ($agentResult['agent_summary'] ?? '') . "\n";

    if (!empty($agentResult['final_result'])) {
        printResult($agentResult['final_result']);
    }
} else {
    echo "错误：" . ($agentResult['error'] ?? '未知错误') . "\n";
}


// ── 辅助函数 ──────────────────────────────────────────────────────────────────

function printResult(array $result): void
{
    if (!($result['success'] ?? true)) {
        echo "错误：" . ($result['error'] ?? '未知') . "\n";
        return;
    }

    $summary = $result['summary'] ?? [];
    echo sprintf(
        "使用箱子：%d 个，平均利用率：%.1f%%，全部装入：%s\n",
        $summary['total_bins_used'] ?? 0,
        ($summary['avg_utilization'] ?? 0) * 100,
        ($summary['all_placed'] ?? false) ? '是' : '否'
    );

    foreach ($result['packed_bins'] ?? [] as $i => $bin) {
        echo sprintf(
            "\n箱子%d：%s  利用率%.1f%%  总重%.1fkg\n",
            $i + 1, $bin['bin_type'], $bin['utilization'] * 100, $bin['total_weight']
        );
        foreach ($bin['items'] as $item) {
            $p = $item['position'];
            echo sprintf(
                "  %-20s 位置(%.0f,%.0f,%.0f)\n",
                $item['id'], $p['x'], $p['y'], $p['z']
            );
        }
    }

    if (!empty($result['unplaced_items'])) {
        echo "\n未装入货物：" . implode(', ', $result['unplaced_items']) . "\n";
    }
}
